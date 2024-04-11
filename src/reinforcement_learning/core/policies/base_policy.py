import abc

import torch
from torch import nn


class BasePolicy(nn.Module, abc.ABC):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, obs: torch.Tensor):
        return self.network(obs)
