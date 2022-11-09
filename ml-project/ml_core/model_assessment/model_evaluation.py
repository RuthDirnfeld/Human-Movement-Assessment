import time

import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


def get_classification_score(preds, labels):
    return accuracy_score(labels, preds)


def regression_score(preds, labels):
    return mean_squared_error(labels, preds)


def get_response_time(model, input_data):
    response_time = 0
    for i in range(len(input_data)):
        start = time.time()
        model.predict(input_data.iloc[i].to_frame().T)
        response_time += time.time() - start
    return response_time / len(input_data)


class ModelEvaluation:
    def __init__(self, model, get_model_score, feature_extractor, data, inputs=None, labels=None):
        self._model = model
        self._get_model_score = get_model_score
        self._feature_extractor = feature_extractor
        self._data = data
        if inputs is None or labels is None:
            train_data = pd.read_csv(self._data.train, decimal=",")
            test_data = pd.read_csv(self._data.validation, decimal=",")
            x_train = self._feature_extractor.process_features(train_data)
            y_train = self._feature_extractor.process_labels(train_data)
            self._X_test = self._feature_extractor.process_features(test_data)
            self._y_test = self._feature_extractor.process_labels(test_data)
        else:
            x_train, self._X_test, y_train, self._y_test = train_test_split(
                inputs, labels, train_size=0.33, random_state=0)

        self._model = self._model.fit(x_train, y_train)
        self._pred = self._model.predict(self._X_test)

    def get_model_score(self):
        return self._get_model_score(self._y_test, self._pred)

    def get_model_response_time(self):
        return get_response_time(self._model, self._X_test)
