from typing import Optional

import torch
from torch import nn

from src.hyper_parameters import HyperParameters
from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.policies.components.actor import Actor
from src.reinforcement_learning.core.policies.components.feature_extractors import FeatureExtractor
from src.reinforcement_learning.core.policies.components.v_critic import VCritic
from src.reinforcement_learning.core.type_aliases import TensorObs
from src.tags import Tags

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

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'critic': self.critic.collect_hyper_parameters(),
        })

    def collect_tags(self) -> Tags:
        return self.combine_tags(super().collect_tags(), self.critic.collect_tags())

    def forward(self, obs: TensorObs) -> tuple[ActionSelector, torch.Tensor]:
        obs = self.shared_feature_extractor(obs)
        return self.actor(obs), self.critic(obs.detach())

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.shared_feature_extractor(obs)
        return self.critic(obs)
