import numpy as np
import torch
from torch import nn

from src.reinforcement_learning.core.policies.base_policy import BasePolicy


class ActorCriticSTMPolicy(BasePolicy):

    def __init__(self, network: nn.Module):
        super().__init__(network)

    def process_obs(self, obs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_tensor = torch.tensor(obs)
        return self(obs_tensor)

    def predict_value(self, obs: np.ndarray) -> torch.Tensor:
        return self.process_obs(obs)[1]
