import numpy as np
import pickle
import tensorflow as tf
from exercise_performance_scorer.model import ExercisePerformanceModel


class ExercisePerformanceEstimator:
    def __init__(self, exercise_performance_scorer_config, model_path=None):
        self.number_of_frames = 20

        self._config = exercise_performance_scorer_config
        self._model_path = model_path or self._config.model.path
        self._ml_model = ExercisePerformanceModel(self._config)
        self._ml_model.compile()
        self._session = tf.Session()
        self._session.run(tf.global_variables_initializer())
        self._ml_model.load(self._model_path)
        self._feature_processor = self._load_feature_processor()
        self._label_processor = self._load_label_processor()

    def predict(self, input_features):
        input_features = self._feature_processor.transform(input_features)
        predictions = self._ml_model.predict(np.array([input_features]))
        predictions = self._label_processor.inverse_transform(predictions)
        score = predictions[0][0]
        return score

    def _load_label_processor(self):
        with open(self._config.model.label_processor_path, "rb") as f:
            processor = pickle.load(f)
        return processor

    def _load_feature_processor(self):
        with open(self._config.model.feature_processor_path, "rb") as f:
            processor = pickle.load(f)
        return processor
