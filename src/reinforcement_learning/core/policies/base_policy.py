import abc
from typing import Optional

import numpy as np
import torch
import torch.distributions as dist
from torch import nn

TensorOrNpArray = torch.Tensor | np.ndarray


class BasePolicy(nn.Module, abc.ABC):

    def __init__(self, network: nn.Module, continuous_actions: bool, actions_std: float = None):
        super().__init__()
        self.network = network
        self.continuous_actions = continuous_actions

        self.actions_std: Optional[float] = None
        self.set_actions_std(actions_std)


    def set_actions_std(self, actions_std: float):
        if not self.continuous_actions and actions_std is not None:
            raise ValueError(f'actions_std has to be None when using discrete actions ({actions_std = })')
        if self.continuous_actions and (actions_std is None or actions_std <= 0):
            raise ValueError(f'actions_std needs to be positive when using continuous actions ({actions_std = })')

        self.actions_std = actions_std

    def forward(self, obs: torch.Tensor):
        return self.network(obs)

    def create_actions_dist(self, action_logits) -> dist.Distribution:
        if self.continuous_actions:
            return dist.Normal(loc=action_logits, scale=self.actions_std)
        else:
            return dist.Categorical(logits=action_logits)

    def predict_actions(self, obs: TensorOrNpArray) -> dist.Distribution:
        return self.process_obs(obs)[0]

    @abc.abstractmethod
    def process_obs(self, obs: TensorOrNpArray) -> tuple[dist.Distribution, dict[str, torch.Tensor]]:
        raise NotImplemented

    @staticmethod
    def as_tensor(obs: TensorOrNpArray) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs
        return torch.tensor(obs, dtype=torch.float32)
