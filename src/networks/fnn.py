from dataclasses import dataclass
from typing import Callable, Literal

import torch.nn as nn

from .nn_base import NNBase, HyperParameters

FnnNormalizationLocation = Literal['pre-layer', 'pre-activation', 'post-activation', 'post-dropout', None]

@dataclass
class FNNHyperParameters(HyperParameters):
    input_size: int
    hidden_sizes: list[int]
    output_size: int

    activation_provider: Callable[[], nn.Module] = lambda: nn.LeakyReLU()
    activate_last_layer: bool = False

    input_normalization_provider: Callable[[int], nn.Module] = None
    layer_normalization_location: FnnNormalizationLocation = None
    layer_normalization_provider: Callable[[int], nn.Module] = None
    output_normalization_provider: Callable[[int], nn.Module] = None

    dropout_p: float = None
    dropout_last_layer: bool = False

    layer_initialization: Callable[[nn.Linear], None] = lambda l: None


class FNN(NNBase):

    def __init__(
            self,
            input_size: int,
            hidden_sizes: list[int],
            output_size: int,
            activation_provider: Callable[[], nn.Module] = lambda: nn.LeakyReLU(),
            activate_last_layer: bool = False,
            input_normalization_provider: Callable[[int], nn.Module] = None,
            layer_normalization_location: FnnNormalizationLocation = None,
            layer_normalization_provider: Callable[[int], nn.Module] = None,
            output_normalization_provider: Callable[[int], nn.Module] = None,
            dropout_p: float = None,
            dropout_last_layer: bool = False,
            layer_initialization: Callable[[nn.Linear], None] = lambda l: None,
    ):
        super().__init__()

        layers_sizes = [input_size] + hidden_sizes + [output_size]

        layers = []
        for i in range(len(layers_sizes) - 1):
            layer_in_size = layers_sizes[i]
            layer_out_size = layers_sizes[i + 1]
            is_last_layer = i == len(layers_sizes) - 2

            if layer_normalization_location == 'pre-layer':
                layers.append(layer_normalization_provider(layer_in_size))

            linear = nn.Linear(layer_in_size, layer_out_size)
            layer_initialization(linear)
            layers.append(linear)

            if layer_normalization_location == 'pre-activation':
                layers.append(layer_normalization_provider(layer_out_size))

            if not is_last_layer or activate_last_layer:
                layers.append(activation_provider())

            if layer_normalization_location == 'post-activation':
                layers.append(layer_normalization_provider(layer_out_size))

            if NNBase.is_dropout_active(dropout_p) and (not is_last_layer or dropout_last_layer):
                layers.append(nn.Dropout(dropout_p))

            if layer_normalization_location == 'post-dropout':
                layers.append(layer_normalization_provider(layer_out_size))

        if input_normalization_provider is not None:
            layers.insert(0, input_normalization_provider(input_size))

        if output_normalization_provider is not None:
            layers.append(output_normalization_provider(output_size))

        self.fnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fnn(x)
        return x
