from typing import Callable
from dataclasses import dataclass

import torch.nn as nn

from .fnn import FNN, FNNHyperParameters, FnnNormalizationLocation


def lecun_initialization(linear: nn.Linear):
    nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='linear')
    nn.init.zeros_(linear.bias)


@dataclass(init=False)
class SelfNormalizingFNNHyperParameters(FNNHyperParameters):
    normalize_input: bool = False
    
    def __init__(
            self,
            input_size: int,
            hidden_sizes: list[int],
            output_size: int,
            activate_last_layer: bool = False,
            input_normalization_provider: Callable[[int], nn.Module] = None,
            output_normalization_provider: Callable[[int], nn.Module] = None,
    ):
        super().__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,

            activation_provider=lambda: nn.SELU(),
            activate_last_layer=activate_last_layer,

            input_normalization_provider=input_normalization_provider,
            layer_normalization_location=None,
            layer_normalization_provider=None,
            output_normalization_provider=output_normalization_provider,

            dropout_p=0.0,
            dropout_last_layer=False,

            layer_initialization=lecun_initialization,
        )


class SelfNormalizingFNN(FNN):

    def __init__(
            self,
            input_size: int,
            hidden_sizes: list[int],
            output_size: int,
            activate_last_layer: bool = False,
            input_normalization_provider: Callable[[int], nn.Module] = None,
            output_normalization_provider: Callable[[int], nn.Module] = None,
    ):
        super().__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,

            activation_provider=lambda: nn.SELU(),
            activate_last_layer=activate_last_layer,

            input_normalization_provider=input_normalization_provider,
            layer_normalization_location=None,
            layer_normalization_provider=None,
            output_normalization_provider=output_normalization_provider,

            dropout_p=0.0,
            dropout_last_layer=False,

            layer_initialization=lecun_initialization,
        )

    def forward(self, x):
        x = self.fnn(x)
        return x
