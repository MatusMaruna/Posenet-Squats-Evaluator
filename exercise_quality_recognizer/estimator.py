import pickle
import tensorflow as tf
from exercise_quality_recognizer.model import ExerciseQualityModel
import os


class ExerciseQualityEstimator:
    def __init__(self, exercise_performance_scorer_config, model_path=None):
        self._config = exercise_performance_scorer_config
        self._model_path = model_path or self._config.model.path
        self._ml_model = ExerciseQualityModel(self._config)
        self._ml_model.compile()
        self._session = tf.Session()
        self._session.run(tf.global_variables_initializer())
        self._ml_model.load(self._model_path)
        self._feature_processor = self._load_feature_processor()

    def predict(self, input_features):
        input_features = self._feature_processor.transform(input_features)
        predictions = self._ml_model.predict_classes(input_features)
        predictions = self._label_processor.inverse_transform(predictions)
        return predictions

    def _load_feature_processor(self):
        processors = []
        for i in range(9):
            with open(os.path.splitext(self._config.model.feature_processor_path)[0] + '_' + str(i)
                      + os.path.splitext(self._config.model.feature_processor_path)[1], "rb") as f:
                processors.append(pickle.load(f))
        return processors
