from dataclasses import dataclass

import torch
from torch import nn

from src.networks.nn_base import NNBase, HyperParameters


@dataclass
class ResidualConnectionHyperParameters(HyperParameters):
    module: nn.Module
    num_features: int
    residual_weight: float = 1.0
    affine_residual_weight: bool = False


class ResidualConnection(NNBase):

    def __init__(
            self,
            module_provider: nn.Module,
            num_features: int,
            residual_weight=1.0,
            affine_residual_weight=False
    ):
        super().__init__()

        self.module = module_provider()
        self.weight = nn.Parameter(torch.ones(num_features).float() * residual_weight,
                                   requires_grad=affine_residual_weight)

    def forward(self, x: torch.Tensor):
        return self.module(x) + self.weight * x





