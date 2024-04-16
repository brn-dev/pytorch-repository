from overrides import override

import numpy as np
import torch
from torch import nn
import torch.distributions as dist

from src.reinforcement_learning.core.policies.actor_critic_policy import ActorCriticPolicy, VALUE_ESTIMATES_KEY

STATE_PREDS_KEY = 'state_preds'


class ActorCriticSTMPolicy(ActorCriticPolicy):

    def __init__(self, network: nn.Module, continuous_actions: bool, actions_std: float = None):
        super().__init__(network, continuous_actions, actions_std)

    @override
    def process_obs(self, obs: np.ndarray) -> tuple[dist.Distribution, dict[str, torch.Tensor]]:
        action_logits, value_estimates, state_preds = self.__predict_actions_values_and_states(obs)
        actions_dist = self.create_actions_dist(action_logits)
        return actions_dist, {VALUE_ESTIMATES_KEY: value_estimates, STATE_PREDS_KEY: state_preds}

    @override
    def predict_values(self, obs: np.ndarray) -> torch.Tensor:
        return self.__predict_actions_values_and_states(obs)[1]

    def predict_values_and_states(self, obs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        actions, value_estimates, state_preds = self.__predict_actions_values_and_states(obs)
        return value_estimates, state_preds

    def __predict_actions_values_and_states(self, obs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_tensor = torch.tensor(obs)
        action_logits, value_estimates, state_preds = self(obs_tensor)
        return action_logits, value_estimates, state_preds
