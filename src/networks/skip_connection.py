from typing import Callable

import torch
from torch import nn

from src.networks.nn_base import NNBase
from src.networks.weighing import WeighingBase


class SkipConnection(NNBase):

    def __init__(
            self,
            module: nn.Module,
            num_features: int,

            skip_connection_weight: float | torch.Tensor | WeighingBase | Callable[[int], WeighingBase] = 1.0,
            skip_connection_weight_affine: bool = False,

            module_out_weight: float | torch.Tensor | WeighingBase | Callable[[int], WeighingBase] = 1.0,
            module_out_weight_affine: bool = False,

            dropout_p: float = 0.0,
            normalization_provider: Callable[[int], nn.Module] = None,
    ):
        super().__init__()

        self.module = module

        self.weigh_skip_connection = \
            WeighingBase.to_weight(num_features, skip_connection_weight, skip_connection_weight_affine)
        self.weigh_module_out = \
            WeighingBase.to_weight(num_features, module_out_weight, module_out_weight_affine)

        self.dropout = NNBase.create_dropout(dropout_p) or nn.Identity()
        self.norm = NNBase.provide(normalization_provider, num_features) or nn.Identity()


    def forward(self, x, *args, **kwargs):
        return self.norm(self.dropout(self.weigh_module_out(self.module(x, *args, **kwargs)) + self.weigh_skip(x)))
