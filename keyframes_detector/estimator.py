from enum import Enum
import numpy as np
import pickle
import tensorflow as tf
from keyframes_detector.model import KeyFramesDetectorModel


class KeYFrameType(Enum):
    StartFrame = "start_frame"
    StopFrame = "stop_frame"
    BasicFrame = "basic_frame"


class KeyFramesDetectorEstimator:
    def __init__(self, frame_detector_config, model_path=None):
        self._config = frame_detector_config
        self._model_path = model_path or self._config.model.path
        self._ml_model = KeyFramesDetectorModel(self._config)
        self._ml_model.compile()
        self._session = tf.Session()
        self._session.run(tf.global_variables_initializer())
        self._ml_model.load(self._model_path)
        self._feature_processor = self._load_feature_processor()

    def predict(self, input_features):
        input_features = self._feature_processor.transform(input_features)
        predictions = self._ml_model.predict(input_features)
        predictions = np.argmax(predictions, axis=1)
        predictions = [self._get_prediction_type(p) for p in predictions]
        return predictions

    def predict_video_key_frames(self, video_kinect_df, fps, min_video_len_sec=5):
        key_frames_preds = self.predict(video_kinect_df.to_numpy())
        start_video_indexes = np.argwhere(np.array(key_frames_preds) == KeYFrameType.StartFrame).flatten()
        end_video_indexes = np.argwhere(np.array(key_frames_preds) == KeYFrameType.StopFrame).flatten()
        start_frame = min(start_video_indexes) if any(start_video_indexes) else 0
        end_frame = max(end_video_indexes) if any(end_video_indexes) else len(video_kinect_df) - 1

        min_video_frames = int(fps * min_video_len_sec)
        if end_frame <= start_frame or (end_frame - start_frame) < min_video_frames:
            end_frame = len(video_kinect_df) - 1
            start_frame_pred = self._get_last_key_frame_from_group(start_frame, start_video_indexes)
            if (end_frame - start_frame_pred) >= min_video_frames:
                start_frame = start_frame_pred
        return int(start_frame), int(end_frame)

    def _load_feature_processor(self):
        with open(self._config.model.feature_processor_path, "rb") as f:
            processor = pickle.load(f)
        return processor

    def _load_label_processor(self):
        with open(self._config.model.label_processor_path, "rb") as f:
            processor = pickle.load(f)
        return processor

    def _get_prediction_type(self, predictin_code):
        if predictin_code == 2:
            prediction = KeYFrameType.StartFrame
        elif predictin_code == 1:
            prediction = KeYFrameType.StopFrame
        else:
            prediction = KeYFrameType.BasicFrame
        return prediction

    def _get_last_key_frame_from_group(self, initial_key_frame, start_video_indexes):
        while initial_key_frame + 1 in start_video_indexes:
            initial_key_frame = initial_key_frame + 1
        return initial_key_frame
