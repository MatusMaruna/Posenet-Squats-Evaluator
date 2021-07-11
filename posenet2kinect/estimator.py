import pickle
import tensorflow as tf
from posenet2kinect.model import Pose2KinectModel


class Pose2KinectEstimator:
    def __init__(self, pose2kinect_config, model_path=None):
        self._config = pose2kinect_config
        self._model_path = model_path or self._config.model.path
        self._ml_model = Pose2KinectModel(self._config)
        self._ml_model.compile()
        self._session = tf.Session()
        self._session.run(tf.global_variables_initializer())
        self._ml_model.load(self._model_path)
        self._feature_processor = self._load_feature_processor()
        self._label_processor = self._load_label_processor()

    def predict(self, input_features):
        input_features = self._feature_processor.transform(input_features)
        predictions = self._ml_model.predict(input_features)
        predictions = self._label_processor.inverse_transform(predictions)
        return predictions

    def _load_feature_processor(self):
        with open(self._config.model.feature_processor_path, "rb") as f:
            processor = pickle.load(f)
        return processor

    def _load_label_processor(self):
        with open(self._config.model.label_processor_path, "rb") as f:
            processor = pickle.load(f)
        return processor
