from overrides import override

import numpy as np
import torch
import torch.distributions as dist
from torch import nn

from src.reinforcement_learning.core.policies.base_policy import BasePolicy, TensorOrNpArray

VALUE_ESTIMATES_KEY = 'value_estimates'


class ActorCriticPolicy(BasePolicy):

    def __init__(self, network: nn.Module, continuous_actions: bool, actions_std: float = None):
        super().__init__(network, continuous_actions, actions_std)

    @override
    def process_obs(self, obs: TensorOrNpArray) -> tuple[dist.Distribution, dict[str, torch.Tensor]]:
        action_logits, value_estimates = self.predict_actions_and_values(obs)
        return self.create_actions_dist(action_logits), {VALUE_ESTIMATES_KEY: value_estimates}

    def predict_values(self, obs: TensorOrNpArray) -> torch.Tensor:
        return self.predict_actions_and_values(obs)[1]

    def predict_actions_and_values(self, obs: TensorOrNpArray) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = self.as_tensor(obs)
        actions_logits, value_estimates = self(obs_tensor)
        return actions_logits, value_estimates
