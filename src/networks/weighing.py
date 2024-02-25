import abc
from typing import Callable

import torch
from torch import nn

from src.networks.nn_base import NNBase


class WeighingBase(NNBase, abc.ABC):

    @abc.abstractmethod
    def weigh(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplemented

    def forward(self, x: torch.Tensor):
        return self.weigh(x)

    @staticmethod
    def to_weight(
            num_features: int = None,
            weight: float | torch.Tensor | "WeighingBase" | Callable[[int], "WeighingBase"] = 1.0,
            affine: bool = False,
    ):
        if isinstance(weight, float) or isinstance(weight, torch.Tensor):
            return BasicWeighing(num_features, weight, affine)
        if isinstance(weight, WeighingBase):
            return weight
        if isinstance(weight, Callable):
            return weight(num_features)


class BasicWeighing(WeighingBase):

    def __init__(
            self,
            num_features: int = None,
            initial_value: float | torch.Tensor = 1.0,
            affine: bool = False,
    ):
        super().__init__()

        assert isinstance(initial_value, torch.Tensor) or num_features is not None

        if isinstance(initial_value, float):
            initial_value = torch.ones(num_features) * initial_value

        self.w = nn.Parameter(initial_value, requires_grad=affine)

    def weigh(self, x: torch.Tensor):
        return self.w * x


class ModuleWeighing(WeighingBase):

    def __init__(
            self,
            module: nn.Module,
    ):
        super().__init__()
        self.module = module

    def weigh(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x) * x
