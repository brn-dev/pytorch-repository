import abc
from typing import Optional, Callable

import numpy as np
import torch
import torch.distributions as dist
from torch import nn

TensorOrNpArray = torch.Tensor | np.ndarray


class BasePolicy(nn.Module, abc.ABC):

    def __init__(self, network: nn.Module, action_dist_provider: Callable[[torch.Tensor], dist.Distribution]):
        super().__init__()
        self.network = network
        self.action_dist_provider = action_dist_provider

    def forward(self, obs: torch.Tensor):
        return self.network(obs)

    def predict_actions(self, obs: TensorOrNpArray) -> dist.Distribution:
        return self.process_obs(obs)[0]

    @abc.abstractmethod
    def process_obs(self, obs: TensorOrNpArray) -> tuple[dist.Distribution, dict[str, torch.Tensor]]:
        raise NotImplemented
