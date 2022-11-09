import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from config import general_config
from logger import setup_logger
from ml_core.expert_score_regressor.feature_extractor import FeatureExtractor
from ml_core.model_assessment.model_evaluation import regression_score, get_response_time


class ExpertScoreModelTrainer:
    def __init__(self, data_path):
        self._data_path = data_path
        self._feature_extractor = FeatureExtractor()
        self._logger = setup_logger("Regressor Logger")

    def process_dataset(self, data):
        # Shuffle data
        data = data.sample(frac=1).reset_index(drop=True)
        return data

    def split_dataset(self, recreate_data):
        if recreate_data:
            data = pd.read_csv(self._data_path, decimal=",")
            data = self.process_dataset(data)
            train_set, validation_set = train_test_split(data, test_size=0.3)
            train_set.to_csv(general_config.data.score_expert.train, index=False,
                             decimal=",")
            validation_set.to_csv(general_config.data.score_expert.validation, index=False,
                                  decimal=",")
        else:
            train_set = pd.read_csv(general_config.data.score_expert.train, decimal=",")
            validation_set = pd.read_csv(general_config.data.score_expert.validation,
                                         decimal=",")

        x_train = self._feature_extractor.process_features(train_set)
        y_train = self._feature_extractor.process_labels(train_set)
        x_valid = self._feature_extractor.process_features(validation_set)
        y_valid = self._feature_extractor.process_labels(validation_set)
        return x_train, y_train, x_valid, y_valid

    def get_cross_validation_score(self, model, x_data, y_data, cv=4):
        scorer = make_scorer(regression_score)
        scores = cross_val_score(model, x_data, y_data, cv=cv, scoring=scorer)
        return np.mean(scores)

    def train(self, recreate_data=False, cross_validation=False):
        x_train, y_train, x_valid, y_valid = self.split_dataset(recreate_data)
        model = RandomForestRegressor(max_features=11, random_state=1)
        if cross_validation:
            cross_valid_score = self.get_cross_validation_score(
                model, pd.concat([x_train, x_valid]), pd.concat([y_train, y_valid]))
            self._logger.info('Cross Validation Accuracy Score' + ': %f', cross_valid_score)
        model.fit(x_train, y_train)

        train_preds = model.predict(x_train)
        valid_preds = model.predict(x_valid)
        self._logger.info('MSE Train' + ': %f', regression_score(y_train, train_preds))
        self._logger.info('MSE Test' + ': %f', regression_score(y_valid, valid_preds))
        self._logger.info('Mean Model Response Time : %fs', get_response_time(model, x_train))
        self._save(model)

    def _save(self, model):
        with open(general_config.ml.expert_score_model_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    dataset_path = "../../data/datasets/AimoScore_WeakLink_big_scores.csv"
    trainer = ExpertScoreModelTrainer(dataset_path)
    trainer.train(recreate_data=True, cross_validation=True)
