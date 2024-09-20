from src.reinforcement_learning.core.policies.components.base_module import BaseModule
from src.reinforcement_learning.core.policies.components.feature_extractors import FeatureExtractor


class BasePolicyComponent(BaseModule):

    def __init__(self, feature_extractor: FeatureExtractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_extractor.set_train_mode(False)
