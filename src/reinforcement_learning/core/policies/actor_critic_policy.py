from overrides import override

import numpy as np
import torch
from torch import nn

from src.reinforcement_learning.core.policies.base_policy import BasePolicy


VALUE_ESTIMATES_KEY = 'value_estimates'

class ActorCriticPolicy(BasePolicy):

    def __init__(self, network: nn.Module):
        super().__init__(network)

    @override
    def process_obs(self, obs: np.ndarray) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        actions, value_estimates = self.__predict_actions_and_values(obs)
        return actions, {VALUE_ESTIMATES_KEY: value_estimates}

    def predict_values(self, obs: np.ndarray) -> torch.Tensor:
        return self.__predict_actions_and_values(obs)[1]

    def __predict_actions_and_values(self, obs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = torch.tensor(obs)
        actions_logits, value_estimates = self(obs_tensor)
        return actions_logits, value_estimates
