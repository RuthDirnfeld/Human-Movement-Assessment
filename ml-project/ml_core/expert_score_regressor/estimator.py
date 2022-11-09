import pickle

from config import general_config
from ml_core.expert_score_regressor.feature_extractor import FeatureExtractor


class ExpertScoreModelEstimator:
    def __init__(self):
        self._model = self.load()
        self._feature_extractor = FeatureExtractor()

    def eval(self, data):
        features = self._feature_extractor.process_features(data)
        print(self._model.feature_importances_)
        return self._model.predict(features)

    def load(self):
        with open(general_config.ml.expert_score_model_path, "rb") as f:
            return pickle.load(f)
