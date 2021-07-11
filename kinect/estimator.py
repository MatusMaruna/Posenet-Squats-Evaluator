import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from kinect.model import KinectModel


class KinectEstimator:
    def __init__(self, kinect_config, model_path=None):
        self._config = kinect_config
        self._model_path = model_path or self._config.model.path
        self._ml_model = KinectModel(self._config)
        self._ml_model.compile()
        self._session = tf.Session()
        self._session.run(tf.global_variables_initializer())
        self._ml_model.load(self._model_path)
        self._feature_processor = self._load_feature_processor()
        self._label_processor = self._load_label_processor()
        self._columns = ['head_x', 'head_y', 'head_z', 'left_shoulder_x', 'left_shoulder_y',
                         'left_shoulder_z', 'left_elbow_x', 'left_elbow_y', 'left_elbow_z', 'right_shoulder_x',
                         'right_shoulder_y', 'right_shoulder_z', 'right_elbow_x', 'right_elbow_y', 'right_elbow_z',
                         'left_hand_x', 'left_hand_y', 'left_hand_z', 'right_hand_x', 'right_hand_y', 'right_hand_z',
                         'left_hip_x', 'left_hip_y', 'left_hip_z', 'right_hip_x', 'right_hip_y', 'right_hip_z',
                         'left_knee_x', 'left_knee_y', 'left_knee_z', 'right_knee_x', 'right_knee_y', 'right_knee_z',
                         'left_foot_x', 'left_foot_y', 'left_foot_z', 'right_foot_x', 'right_foot_y', 'right_foot_z']

    def predict(self, input_features):
        input_features = self._feature_processor.transform(input_features)
        predictions = self._ml_model.predict(input_features)
        predictions = self._label_processor.inverse_transform(predictions)
        return predictions

    def predict3d(self, input_features, return_df=False):
        predictions_z = self.predict(input_features)
        features_3d = []
        for x_y, z in zip(input_features, predictions_z):
            record_3d = self.get_3d_record(x_y, z)
            features_3d.append(record_3d)

        if return_df:
            features_3d = pd.DataFrame(data=features_3d, columns=self._columns)
        else:
            features_3d = np.array(features_3d)
        return features_3d

    def get_3d_record(self, x_y_features, z_features):
        record_3d = []
        for index, z in enumerate(z_features):
            x_y = x_y_features[(index * 2):((index * 2) + 2)]
            x_y_z = list(x_y) + [z]
            record_3d += x_y_z
        return record_3d

    def _load_feature_processor(self):
        with open(self._config.model.feature_processor_path, "rb") as f:
            processor = pickle.load(f)
        return processor

    def _load_label_processor(self):
        with open(self._config.model.label_processor_path, "rb") as f:
            processor = pickle.load(f)
        return processor
