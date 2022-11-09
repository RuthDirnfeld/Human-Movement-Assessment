import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from config import general_config
from logger import setup_logger
from ml_core.model_assessment.model_evaluation import get_classification_score, get_response_time
from ml_core.weak_link_classifier.feature_extractor import FeatureExtractor


class WeakLinkModelTrainer:
    def __init__(self, data_path):
        self._feature_extractor = FeatureExtractor()
        self._data_path = data_path
        self._logger = setup_logger("Classifier Logger")

    def process_dataset(self, data):
        # Shuffle data
        data = data.sample(frac=1).reset_index(drop=True)
        return data

    def split_dataset(self, recreate_data):
        if recreate_data:
            data = pd.read_csv(self._data_path, decimal=",")
            data = self.process_dataset(data)
            train_set, validation_set = train_test_split(data, test_size=0.3)
            train_set.to_csv(general_config.data.weak_link.train, index=False,
                             decimal=",")
            validation_set.to_csv(general_config.data.weak_link.validation, index=False,
                                  decimal=",")
        else:
            train_set = pd.read_csv(general_config.data.weak_link.train, decimal=",")
            validation_set = pd.read_csv(general_config.data.weak_link.validation,
                                         decimal=",")

        x_train = self._feature_extractor.process_features(train_set)
        y_train = self._feature_extractor.process_labels(train_set)
        x_valid = self._feature_extractor.process_features(validation_set)
        y_valid = self._feature_extractor.process_labels(validation_set)
        return x_train, y_train, x_valid, y_valid

    def get_cross_validation_score(self, model, x_data, y_data, cv=4):
        scores = cross_val_score(model, x_data, y_data, cv=cv)
        return np.mean(scores)

    def get_grid_searched_model(self, x_data, y_data):
        scorer = make_scorer(get_classification_score)
        gsc = GridSearchCV(
            estimator=SVC(),
            param_grid={'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                        'C': [0.01, 0.1, 1, 10, 100],
                        'gamma': [0.5, 1, 2, 3, 4]
                        },
            cv=4, scoring=scorer, verbose=0, n_jobs=-1)
        grid_result = gsc.fit(x_data, y_data)
        best_params = grid_result.best_params_
        self._logger.info(f"Best parameters are: {best_params}")
        model = SVC(kernel=best_params["kernel"], C=best_params["C"], gamma=best_params["gamma"])
        return model

    def train(self, recreate_data=False, cross_validation=False, grid_search=False):
        x_train, y_train, x_valid, y_valid = self.split_dataset(recreate_data)

        if grid_search:
            model = self.get_grid_searched_model(
                pd.concat([x_train, x_valid]), pd.concat([y_train, y_valid]))
        else:
            model = SVC(kernel="poly", C=0.1, gamma=0.5)

        if cross_validation:
            cross_valid_score = self.get_cross_validation_score(
                model, pd.concat([x_train, x_valid]), pd.concat([y_train, y_valid]))
            self._logger.info('Cross Validation Accuracy Score' + ': %f', cross_valid_score)
        model.fit(x_train, y_train)

        train_preds = model.predict(x_train)
        valid_preds = model.predict(x_valid)
        self._logger.info('Train Accuracy Score' + ': %f', get_classification_score(y_train, train_preds))
        self._logger.info('Test Accuracy Score' + ': %f', get_classification_score(y_valid, valid_preds))
        self._logger.info('Mean Model Response Time : %fs', get_response_time(model, x_train))
        self._save(model)

    def _save(self, model):
        with open(general_config.ml.weak_link_model_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    dataset_path = "../../data/datasets/AimoScore_WeakLink_big_scores_labels.csv"
    trainer = WeakLinkModelTrainer(dataset_path)
    trainer.train(recreate_data=True, cross_validation=True, grid_search=True)
