import tensorflow as tf
from frame_quality_recognizer.model import FrameQualityRecognizerModel


class FrameQualityRecognizerEstimator:
    def __init__(self, frame_detector_config, model_path=None):
        self._config = frame_detector_config
        self._model_path = model_path or self._config.model.path
        self._ml_model = FrameQualityRecognizerModel(self._config)
        self._ml_model.compile()
        self._session = tf.Session()
        self._session.run(tf.global_variables_initializer())
        self._ml_model.load(self._model_path)

    def predict(self, input_features):
        predictions = self._ml_model.predict(input_features)
        return predictions
