import unittest

import pandas as pd
from sklearn.metrics import mean_squared_error

from config import general_config
from ml_core.expert_score_regressor.estimator import ExpertScoreModelEstimator


class AccuracyTest(unittest.TestCase):
    def setUp(self):
        self.me = ExpertScoreModelEstimator()
        self.me.load()
        self.df = pd.read_csv(general_config.data.score_expert.validation, decimal=',')
        self.df_aimo = self.df['AimoScore']

    def test_mse_threshold(self):
        val_preds = self.me.eval(self.df)
        score = mean_squared_error(list(self.df_aimo), list(val_preds))
        self.assertTrue(score < general_config.ml.threshold)


if __name__ == '__main__':
    unittest.main()
