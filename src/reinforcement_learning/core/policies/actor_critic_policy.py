from typing import Optional

import torch
from torch import nn

from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.policies.components.actor import Actor
from src.reinforcement_learning.core.policies.components.feature_extractors import FeatureExtractor
from src.reinforcement_learning.core.policies.components.v_critic import VCritic
from src.reinforcement_learning.core.type_aliases import TensorObs

VALUE_ESTIMATES_KEY = 'value_estimates'


class ActorCriticPolicy(BasePolicy):

    def __init__(
            self,
            actor: Actor,
            critic: VCritic | nn.Module,  # A state value critic is essentially just a module
            shared_feature_extractor: Optional[FeatureExtractor] = None,
    ):
        super().__init__(
            actor=actor,
            shared_feature_extractor=shared_feature_extractor
        )
        self.critic = critic

    def forward(self, obs: TensorObs) -> tuple[ActionSelector, torch.Tensor]:
        obs = self.shared_feature_extractor(obs)
        return self.actor(obs), self.critic(obs.detach())

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.shared_feature_extractor(obs)
        return self.critic(obs)
