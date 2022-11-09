import pickle
import tensorflow as tf
from exercise_quality_recognizer.model import ExerciseQualityModel
import os


class ExerciseQualityEstimator:
    def __init__(self, exercise_performance_scorer_config, model_path=None):
        self.number_of_frames = 10

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
        input_features = input_features.reshape((1, input_features.shape[0], input_features.shape[1], 1))
        predictions = self._ml_model.predict(input_features)[0][0]
        return predictions

    def _load_feature_processor(self):
        with open(self._config.model.feature_processor_path, "rb") as f:
            processor = pickle.load(f)
        return processor
