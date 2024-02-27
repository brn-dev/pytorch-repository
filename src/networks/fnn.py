from typing import Callable

import torch.nn as nn

from src.networks.init import lecun_initialization
from src.networks.nn_base import NNBase


class FNN(NNBase):

    def __init__(
            self,
            input_size: int,
            hidden_sizes: list[int],
            output_size: int,
            component_order: str = 'LADN',  # L=Linear, A=Activation, D=Dropout(if>0), N=Normalization(if provided)
            layer_initialization: Callable[[nn.Linear], None] = lambda l: None,
            activation_provider: NNBase.Provider = lambda: nn.LeakyReLU(),
            activate_last_layer: bool = False,
            dropout_p: float = None,
            dropout_last_layer: bool = False,
            dropout_input: bool = False,
            dropout_output: bool = False,
            normalization_provider_layer: NNBase.Provider = None,
            normalize_last_layer=False,
            normalization_provider_input: NNBase.Provider = None,
            normalization_provider_output: NNBase.Provider = None,
    ):
        super().__init__()
        print(input_size, hidden_sizes, output_size)

        component_order = component_order.upper()
        assert component_order.count('L') == 1, 'There has to exactly one Linear module per sublayer'
        assert not normalize_last_layer or normalization_provider_layer is not None, \
            'Provide a normalization when normalize_last_layer == True'

        self.layers: list[nn.Sequential | nn.Module] = []
        layers_sizes = [input_size] + hidden_sizes + [output_size]
        layer_in_size, layer_out_size, current_size, in_last_layer = -1, -1, -1, False
        layer_components: list[nn.Module] = []

        def create_linear():
            linear = nn.Linear(layer_in_size, layer_out_size)
            layer_initialization(linear)
            return linear

        def create_and_append_layer():
            self.layers.append(nn.Sequential(*layer_components))
            del layer_components[:]

        component_create = {
            'L': create_linear,
            'A': lambda: NNBase.provide(activation_provider, _if=activate_last_layer or not in_last_layer,),
            'D': lambda: NNBase.provide_dropout(dropout_p, _if=dropout_last_layer or not in_last_layer),
            'N': lambda: NNBase.provide(
                lambda: normalization_provider_layer(current_size),
                _if=normalize_last_layer or normalization_provider_layer is not None and not in_last_layer,

            )
        }

        if dropout_input:
            self.layers.append(NNBase.provide_dropout(dropout_p))
        if normalization_provider_input is not None:
            self.layers.append(NNBase.provide(normalization_provider_input, input_size))

        for i in range(len(layers_sizes) - 1):
            layer_in_size = layers_sizes[i]
            layer_out_size = layers_sizes[i + 1]
            in_last_layer = i == len(layers_sizes) - 2

            current_size = layer_in_size
            for component_id in component_order:
                component = component_create[component_id]()
                if component is not None:
                    layer_components.append(component)

                    if component_id == 'L':
                        current_size = layer_out_size

            create_and_append_layer()

        if dropout_output:
            self.layers.append(NNBase.provide_dropout(dropout_p))
        if normalization_provider_output is not None:
            self.layers.append(NNBase.provide(normalization_provider_output, output_size))

        self.fnn = nn.Sequential(*self.layers)

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
