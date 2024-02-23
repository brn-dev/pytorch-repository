from typing import Callable
from dataclasses import dataclass

import torch.nn as nn

from .fnn import FNN, FNNHyperParameters, FnnNormalizationLocation


@dataclass
class SelfNormalizingFNNHyperParameters(FNNHyperParameters):
    normalize_input: bool = True

    activation_provider: Callable[[], nn.Module] = lambda: nn.SELU()
    activate_last_layer: bool = False

    normalization_location: FnnNormalizationLocation = None
    normalization_provider: Callable[[int], nn.Module] = lambda num_features: nn.Identity

    dropout_p: float = 0.0
    dropout_last_layer: bool = False

    layer_initialization: Callable[[nn.Linear], None] = lambda l: SelfNormalizingFNN.lecun_initialization


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
            layer_initialization=SelfNormalizingFNN.lecun_initialization,
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

    @classmethod
    def from_hyper_parameters(cls, hyper_parameters: SelfNormalizingFNNHyperParameters):
        return SelfNormalizingFNN(
            input_size=hyper_parameters.input_size,
            hidden_sizes=hyper_parameters.hidden_sizes,
            output_size=hyper_parameters.output_size,
            activate_last_layer=hyper_parameters.activate_last_layer,
            normalize_input=hyper_parameters.normalize_input,
        )
