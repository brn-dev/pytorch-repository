from typing import Callable

import torch.nn as nn

from .init import lecun_initialization
from .nn_base import NNBase


class FNN(NNBase):

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
        super().__init__()

        component_order = component_order.upper()
        assert component_order.count('L') == 1, 'There has to exactly one Linear module per sublayer'
        assert not normalize_last_layer or normalization_provider_layer is not None, \
            'Provide a normalization when normalize_last_layer == True'

        components = []

        layers_sizes = [input_size] + hidden_sizes + [output_size]
        layer_in_size, layer_out_size, current_size, in_last_layer = -1, -1, -1, False

        def append_provided_component_if(condition: bool, provider: Callable[[], nn.Module]) -> bool:
            if condition:
                components.append(provider())
            return condition

        def append_dropout_if(condition: bool) -> bool:
            return append_provided_component_if(
                self.is_dropout_active(dropout_p) and condition,
                lambda: nn.Dropout(dropout_p),
            )

        def append_linear():
            linear = nn.Linear(layer_in_size, layer_out_size)
            layer_initialization(linear)
            components.append(linear)

        component_append = {
            'L': append_linear,
            'A': lambda: append_provided_component_if(
                activate_last_layer or not in_last_layer,
                activation_provider
            ),
            'D': lambda: append_dropout_if(dropout_last_layer or not in_last_layer),
            'N': lambda: append_provided_component_if(
                normalize_last_layer or normalization_provider_layer is not None and not in_last_layer,
                lambda: normalization_provider_layer(current_size)
            )
        }

        append_dropout_if(dropout_input)
        append_provided_component_if(
            normalization_provider_input is not None,
            lambda: normalization_provider_input(input_size)
        )

        for i in range(len(layers_sizes) - 1):
            layer_in_size = layers_sizes[i]
            layer_out_size = layers_sizes[i + 1]
            in_last_layer = i == len(layers_sizes) - 2

            current_size = layer_in_size
            for component_id in component_order:
                component_append[component_id]()

                if component_id == 'L':
                    current_size = layer_out_size

        append_dropout_if(dropout_output)
        append_provided_component_if(
            normalization_provider_output is not None,
            lambda: normalization_provider_input(output_size)
        )

        self.fnn = nn.Sequential(*components)

    def forward(self, x):
        x = self.fnn(x)
        return x


    @staticmethod
    def self_normalizing(
            input_size: int,
            hidden_sizes: list[int],
            output_size: int,
            activate_last_layer: bool = False,
            normalization_provider_input: Callable[[int], nn.Module] = None,
            normalization_provider_output: Callable[[int], nn.Module] = None,
    ):
        return FNN(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,

            component_order='LA',
            layer_initialization=lecun_initialization,
            activation_provider=lambda: nn.SELU(),
            activate_last_layer=activate_last_layer,

            normalization_provider_input=normalization_provider_input,
            normalization_provider_output=normalization_provider_output,
        )
