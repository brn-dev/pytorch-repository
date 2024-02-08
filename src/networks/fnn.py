from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from .nn_base import NNBase
from ..hyper_parameters import HyperParameters

@dataclass
class FNNHyperParameters(HyperParameters):
    input_size: int
    hidden_sizes: list[int]
    output_size: int
    activation_provider: Callable[[], nn.Module] = lambda: nn.LeakyReLU()
    layer_initialization: Callable[[nn.Linear], None] = lambda l: None
    activate_last_layer: bool = False


class FNN(NNBase):

    def __init__(
            self,
            input_size: int,
            hidden_sizes: list[int],
            output_size: int,
            activation_provider: Callable[[], nn.Module] = lambda: nn.LeakyReLU(),
            layer_initialization: Callable[[nn.Linear], None] = lambda l: None,
            activate_last_layer=False,
    ):
        super().__init__()

        layers_sizes = [input_size] + hidden_sizes + [output_size]

        layers = []
        for i in range(len(layers_sizes) - 1):
            linear = nn.Linear(layers_sizes[i], layers_sizes[i + 1])
            layer_initialization(linear)
            layers.append(linear)

            if i < len(layers_sizes) - 2 or activate_last_layer:
                layers.append(activation_provider())

        self.fnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fnn.forward(x)
        return x
