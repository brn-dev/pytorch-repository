import abc
from typing import Optional, Callable

import numpy as np
import torch
import torch.distributions as dist
from torch import nn

from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector

TensorOrNpArray = torch.Tensor | np.ndarray
ActionDistStd = float | torch.Tensor
ActionDistProvider = Callable[[torch.Tensor, ActionDistStd | None], dist.Distribution]


class BasePolicy(nn.Module, abc.ABC):

    def __init__(
            self,
            network: nn.Module,
            action_selector: ActionSelector
    ):
        super().__init__()
        self.network = network
        self.action_selector = action_selector

    @abc.abstractmethod
    def process_obs(self, obs: TensorOrNpArray) -> tuple[ActionSelector, dict[str, torch.Tensor]]:
        raise NotImplemented

    def forward(self, obs: torch.Tensor):
        return self.network(obs)
