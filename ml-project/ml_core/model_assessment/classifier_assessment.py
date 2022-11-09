import warnings

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import general_config
from logger import setup_logger
from ml_core.model_assessment.model_evaluation import ModelEvaluation
from ml_core.model_assessment.model_evaluation import get_classification_score
from ml_core.weak_link_classifier.feature_extractor import FeatureExtractor

warnings.filterwarnings("ignore")
dataset_path = general_config.data.weak_link.full
dataset = pd.read_csv(dataset_path)
dataset = dataset[dataset.WeakLink_label != 'LeftHeelRises']
dataset = dataset[dataset.WeakLink_label != 'RightHeelRises']
logger = setup_logger("Classifier Assessment")

knn = KNeighborsClassifier(n_neighbors=3)
qda = QuadraticDiscriminantAnalysis()
lda = LinearDiscriminantAnalysis()
dtc = DecisionTreeClassifier(max_depth=10)
rtc = RandomForestClassifier(max_depth=10, max_features=24)
gbc = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=10, random_state=1)
svm = SVC(C=0.1, gamma=0.5, kernel='poly')


X_features = FeatureExtractor().process_features(dataset)

models = [knn, qda, lda, dtc, rtc, gbc, svm]
X_data = [dataset.iloc[:, 1:39], dataset.iloc[:, 14:41], dataset.iloc[:, 1:41],
          X_features, X_features, X_features,
          FeatureExtractor().process_features_fms_left_removed(dataset)]
y = dataset['WeakLink_label']
i = 0
for model in models:
    model_eval = ModelEvaluation(model, get_classification_score, FeatureExtractor(), general_config.data.weak_link,
                                 X_data[i], y)
    logger.info('Model: %s', model.__class__.__name__)
    logger.info('Accuracy Score' + ': %f', model_eval.get_model_score())
    logger.info('Mean Model Response Time : %fs', model_eval.get_model_response_time())
    i += 1








