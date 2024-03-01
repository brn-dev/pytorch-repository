from typing import Callable

import torch.nn as nn

from src.networks.forward_net import ForwardNet
from src.networks.init import lecun_initialization
from src.networks.net import Net


class FNN(ForwardNet):

    def __init__(
            self,
            layers_in_out_sizes: list[tuple[int, int]] = None,
            layers_sizes: list[int] = None,
            in_size: int = None,
            out_sizes: list[int] = None,
            num_layers: int = None,
            num_features: int = None,
            component_order: str = 'LADN',  # L=Linear, A=Activation, D=Dropout(if>0), N=Normalization(if provided)
            layer_initialization: Callable[[nn.Linear], nn.Linear] = lambda l: None,
            activation_provider: Callable[[], nn.Module] = lambda: nn.LeakyReLU(),
            activate_last_layer: bool = False,
            dropout_p: float = None,
            dropout_last_layer: bool = False,
            normalization_provider_layer: Net.Provider = None,
            normalize_last_layer=False,
    ):

        component_order = component_order.upper()
        assert component_order.count('L') == 1, 'There has to exactly one Linear module per sublayer'
        assert not normalize_last_layer or normalization_provider_layer is not None, \
            'Provide a normalization when normalize_last_layer == True'

        def create_layer(layer_nr, is_last_layer, in_features, out_features):
            component_create = {
                'L': lambda: layer_initialization(nn.Linear(in_features, out_features)),
                'A': lambda: Net.provide(activation_provider, _if=activate_last_layer or not is_last_layer),
                'D': lambda: Net.provide_dropout(dropout_p, _if=dropout_last_layer or not is_last_layer),
                'N': lambda: Net.provide(
                    lambda: normalization_provider_layer(current_size),
                    _if=normalize_last_layer or normalization_provider_layer is not None and not is_last_layer,
                )
            }

            components = []
            current_size = in_features
            for component_id in component_order:
                component = component_create[component_id]()
                if component is not None:
                    components.append(component)

                    if component_id == 'L':
                        current_size = out_features

            return nn.Sequential(*components)

        super().__init__(
            layer_provider=create_layer,
            layers_in_out_sizes=layers_in_out_sizes,
            layers_sizes=layers_sizes,
            in_size=in_size,
            out_sizes=out_sizes,
            num_layers=num_layers,
            num_features=num_features,
        )

    def forward(self, x):
        x = self.fnn(x)
        return x


    @staticmethod
    def self_normalizing(
            layers_in_out_sizes: list[tuple[int, int]] = None,
            layers_sizes: list[int] = None,
            in_size: int = None,
            out_sizes: list[int] = None,
            num_layers: int = None,
            num_features: int = None,
            activate_last_layer: bool = False,
    ):
        return FNN(
            layers_in_out_sizes=layers_in_out_sizes,
            layers_sizes=layers_sizes,
            in_size=in_size,
            out_sizes=out_sizes,
            num_layers=num_layers,
            num_features=num_features,

            component_order='LA',

            layer_initialization=lecun_initialization,
            activation_provider=lambda: nn.SELU(),
            activate_last_layer=activate_last_layer,
        )
