from typing import Optional

import torch
from torch import nn

from src.hyper_parameters import HyperParameters
from src.reinforcement_learning.core.policies.components.base_component import BasePolicyComponent
from src.reinforcement_learning.core.policies.components.feature_extractors import FeatureExtractor, IdentityExtractor


# State value critic
class VCritic(BasePolicyComponent):
    
    def __init__(
            self,
            network: nn.Module,
            feature_extractor: Optional[FeatureExtractor] = None
    ):
        super().__init__(feature_extractor or IdentityExtractor())
        self.network = network

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'network': self.get_hps_or_str(self.network),
        })

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.feature_extractor(obs)
        return self.network(obs)
