from pathlib import Path

from easydict import EasyDict

general_config = EasyDict()
general_config.root = Path(__file__).parent
general_config.ml = EasyDict()
general_config.ml.expert_score_model_path = str(
    general_config.root / "data/models/regressor.pkl")
general_config.ml.weak_link_model_path = str(
    general_config.root / "data/models/classifier.pkl")
general_config.ml.threshold = 0.1
general_config.service = EasyDict()
general_config.service.host = "0.0.0.0"
general_config.service.port = "5000"

general_config.data = EasyDict()
general_config.data.score_expert = EasyDict()
general_config.data.score_expert.train = str(general_config.root / "data/datasets/expert_score_train_set.csv")
general_config.data.score_expert.validation = str(
    general_config.root / "data/datasets/expert_score_validation_set.csv")

general_config.data.weak_link = EasyDict()
general_config.data.weak_link.train = str(general_config.root / "data/datasets/weak_link_train_set.csv")
general_config.data.weak_link.validation = str(
    general_config.root / "data/datasets/weak_link_validation_set.csv")
general_config.data.weak_link.full = str(
    general_config.root / "data/datasets/AimoScore_WeakLink_big_scores_labels.csv")

general_config.logging = EasyDict()
general_config.logging.level = 2
general_config.logging.format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
