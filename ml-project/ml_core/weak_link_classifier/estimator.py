import pickle

from config import general_config
from ml_core.weak_link_classifier.feature_extractor import FeatureExtractor


class WeakLinkModelEstimator:
    def __init__(self):
        self._feature_extractor = FeatureExtractor()
        self._model = self.load()

    def eval(self, data):
        features = self._feature_extractor.process_features(data)
        return self._model.predict(features)

    def load(self):
        with open(general_config.ml.weak_link_model_path, "rb") as f:
            return pickle.load(f)
