from src.hyper_parameters import HyperParameters
from src.reinforcement_learning.core.policies.components.base_rl_module import BaseRLModule
from src.reinforcement_learning.core.policies.components.feature_extractors import FeatureExtractor
from src.tags import Tags


class BasePolicyComponent(BaseRLModule):

    def __init__(self, feature_extractor: FeatureExtractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_extractor.set_train_mode(False)

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'feature_extractor': self.feature_extractor.collect_hyper_parameters(),
        })

    def collect_tags(self) -> Tags:
        return super().collect_tags() + self.feature_extractor.collect_tags()
