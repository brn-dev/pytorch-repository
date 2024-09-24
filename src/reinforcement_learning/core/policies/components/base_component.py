from src.hyper_parameters import HyperParameters
from src.reinforcement_learning.core.policies.components.base_rl_module import BaseRLModule
from src.reinforcement_learning.core.policies.components.feature_extractors import FeatureExtractor


class BasePolicyComponent(BaseRLModule):

    def __init__(self, feature_extractor: FeatureExtractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_extractor.set_train_mode(False)

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'feature_extractor': self.feature_extractor.collect_hyper_parameters(),
        })
