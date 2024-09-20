import torch
from torch import nn

from src.reinforcement_learning.core.policies.components.base_module import BaseModule
from src.reinforcement_learning.core.type_aliases import TensorDict


class FeatureExtractor(BaseModule):
    pass


class IdentityExtractor(FeatureExtractor):

    # noinspection PyMethodMayBeStatic
    def forward(self, x: torch.Tensor):
        return x


class ConcatExtractor(FeatureExtractor):

    def __init__(self, feature_dim: int = -1):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, obs: TensorDict):
        return torch.cat(tuple(v for v in obs.values()), dim=self.feature_dim)
