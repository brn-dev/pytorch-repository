import torch
from torch import nn

from src.networks.nn_base import NNBase


class ResidualConnection(NNBase):

    def __init__(
            self,
            module_provider: NNBase,
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





