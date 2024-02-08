from dataclasses import dataclass

import torch.nn as nn

from .fnn import FNN
from ..hyper_parameters import HyperParameters


@dataclass
class SelfNormalizingFNNHyperParameters(HyperParameters):
    input_size: int
    hidden_sizes: list[int]
    output_size: int

    activate_last_layer: bool = False


class SelfNormalizingFNN(FNN):

    def __init__(
            self,
            input_size: int,
            hidden_sizes: list[int],
            output_size: int,
            activate_last_layer=False,
    ):
        super().__init__(
            input_size,
            hidden_sizes,
            output_size,
            activation_provider=lambda: nn.SELU(),
            activate_last_layer=activate_last_layer,
            normalization_location=None,
            dropout=0.0,
            layer_initialization=self.lecun_initialization,
        )

    def forward(self, x):
        x = self.fnn.forward(x)
        return x

    @staticmethod
    def lecun_initialization(linear: nn.Linear):
        nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(linear.bias)
