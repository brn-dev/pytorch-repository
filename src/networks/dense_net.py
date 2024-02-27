from typing import Callable

import numpy as np
from torch import nn

from src.networks.fnn import FNN
from src.networks.skip_connection import ConcatSkipConnection


class DenseNet(FNN):

    def __init__(
            self,
            input_size: int,
            hidden_sizes: list[int],
            output_size: int,
            component_order: str = 'LADN',  # L=Linear, A=Activation, D=Dropout(if>0), N=Normalization(if provided)
            layer_initialization: Callable[[nn.Linear], None] = lambda l: None,
            activation_provider: Callable[[], nn.Module] = lambda: nn.LeakyReLU(),
            activate_last_layer: bool = False,
            dropout_p: float = None,
            dropout_last_layer: bool = False,
            dropout_input: bool = False,
            dropout_output: bool = False,
            normalization_provider_layer: Callable[[int], nn.Module] = None,
            normalize_last_layer=False,
            normalization_provider_input: Callable[[int], nn.Module] = None,
            normalization_provider_output: Callable[[int], nn.Module] = None,
    ):
        super().__init__(
            **DenseNet.calculate_accumulated_sizes(input_size, hidden_sizes, output_size),
            component_order=component_order,
            layer_initialization=layer_initialization,
            activation_provider=activation_provider,
            activate_last_layer=activate_last_layer,
            dropout_p=dropout_p,
            dropout_last_layer=dropout_last_layer,
            dropout_input=dropout_input,
            dropout_output=dropout_output,
            normalization_provider_layer=normalization_provider_layer,
            normalize_last_layer=normalize_last_layer,
            normalization_provider_input=normalization_provider_input,
            normalization_provider_output=normalization_provider_output,
        )
        self.fnn = nn.Sequential(*[
            ConcatSkipConnection(module) if isinstance(module, nn.Sequential) else module
            for module
            in self.fnn
        ])


    @staticmethod
    def calculate_accumulated_sizes(input_size: int, hidden_sizes: list[int], output_size: int):
        layers_sizes = [input_size] + hidden_sizes + [output_size]
        accumulated_sizes = list(np.cumsum(layers_sizes))
        return {
            'input_size': input_size,
            'hidden_sizes': accumulated_sizes[1:-1],
            'output_size': accumulated_sizes[-1],
        }
