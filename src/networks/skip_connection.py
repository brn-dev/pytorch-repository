from typing import Callable

from torch import nn

from src.networks.nn_base import NNBase
from src.networks.weighing import WeighingBase, BasicWeighing


class SkipConnection(NNBase):

    def __init__(
            self,
            module: nn.Module,
            num_features: int,
            skip_connection_weight: float | WeighingBase | Callable[[int], WeighingBase] = 1.0,
            skip_connection_weight_affine: bool = False,
            dropout_p: float = 0.0,
            normalization_provider: Callable[[int], nn.Module] = None,
    ):
        super().__init__()

        self.module = module

        if skip_connection_weight is None:
            self.weigh = BasicWeighing(
                num_features, initial_value=1.0, affine=skip_connection_weight_affine
            )
        elif isinstance(skip_connection_weight, float):
            self.weigh = BasicWeighing(
                num_features, initial_value=skip_connection_weight, affine=skip_connection_weight_affine
            )
        elif isinstance(skip_connection_weight, WeighingBase):
            self.weigh = skip_connection_weight
        elif isinstance(skip_connection_weight, Callable):
            self.weigh = skip_connection_weight(num_features)

        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.norm = normalization_provider(num_features) if normalization_provider is not None else nn.Identity()


    def forward(self, x, *args, **kwargs):
        return self.norm(self.dropout(self.module(x, *args, **kwargs) + self.weigh(x)))
