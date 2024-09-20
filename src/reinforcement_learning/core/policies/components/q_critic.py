from typing import Callable, Optional

import torch
from torch import nn

from src.reinforcement_learning.core.policies.components.base_component import BasePolicyComponent
from src.reinforcement_learning.core.policies.components.feature_extractors import FeatureExtractor, IdentityExtractor


# State-action value critic
class QCritic(BasePolicyComponent):

    def __init__(
            self,
            n_critics: int,
            create_q_network: Callable[[], nn.Module],
            feature_extractor: Optional[FeatureExtractor] = None,
            feature_dim: int = -1
    ):
        super().__init__(feature_extractor or IdentityExtractor())

        self.n_critics = n_critics

        self.q_networks = nn.ModuleList()
        for _ in range(n_critics):
            q_net = create_q_network()
            self.q_networks.append(q_net)

        self.feature_dim = feature_dim

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        obs = self.feature_extractor(obs)
        q_network_input = torch.cat((obs, actions), dim=self.feature_dim)
        return tuple(q_net(q_network_input) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        obs = self.feature_extractor(obs)
        q_network_input = torch.cat((obs, actions), dim=self.feature_dim)
        return self.q_networks[0](q_network_input)
