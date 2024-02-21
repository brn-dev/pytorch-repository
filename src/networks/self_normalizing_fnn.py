from dataclasses import dataclass

import torch.nn as nn

from .fnn import FNN, HyperParameters


@dataclass
class SelfNormalizingFNNHyperParameters(HyperParameters):
    input_size: int
    hidden_sizes: list[int]
    output_size: int

    activate_last_layer: bool = False
    normalize_input: bool = True


class SelfNormalizingFNN(FNN):

    def __init__(
            self,
            input_size: int,
            hidden_sizes: list[int],
            output_size: int,
            activate_last_layer=False,
            normalize_input=True
    ):
        super().__init__(
            input_size,
            hidden_sizes,
            output_size,
            activation_provider=lambda: nn.SELU(),
            activate_last_layer=activate_last_layer,
            normalization_location=None,
            dropout_p=0.0,
            layer_initialization=self.lecun_initialization,
        )

        if normalize_input:
            self.fnn.insert(0, nn.LayerNorm(input_size))

    def forward(self, x):
        x = self.fnn.forward(x)
        return x

    @staticmethod
    def lecun_initialization(linear: nn.Linear):
        nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(linear.bias)
