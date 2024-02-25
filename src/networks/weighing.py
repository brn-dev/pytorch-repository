import abc

import torch
from torch import nn

from src.networks.nn_base import NNBase


class WeighingBase(abc.ABC, NNBase):

    @abc.abstractmethod
    def weigh(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplemented

    def forward(self, x: torch.Tensor):
        return self.weigh(x)


class BasicWeighing(WeighingBase):

    def __init__(
            self,
            num_features: int,
            initial_value: float,
            affine: bool = False,
    ):
        super().__init__()
        self.w = nn.Parameter(
            torch.ones(num_features) * initial_value,
            requires_grad=affine
        )

    def weigh(self, x: torch.Tensor):
        return self.w * x


class ModuleWeighing(WeighingBase):

    def __init__(
            self,
            module: nn.Module,
    ):
        self.module = module

    def weigh(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x) * x
