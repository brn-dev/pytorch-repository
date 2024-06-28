from typing import Callable

import numpy as np
import torch
from overrides import override

from src.networks.core.layer_connections import LayerConnections
from src.networks.core.layered_net import LayeredNet, LayerProvider
from src.networks.core.net import Net
from src.networks.core.net_list import NetListLike
from src.networks.core.seq_net import SeqNet
from src.networks.core.tensor_shape import TensorShape


class ModulatedDenseSkipNet(LayeredNet):

    def __init__(
            self,
            layers: NetListLike,
            connection_modulators: list[list[Net | None]]
    ):
        super().__init__(
            layers=layers,
            layer_connections=[
                (from_idx, to_idx)
                for to_idx in range(len(layers) + 1)
                for from_idx in range(to_idx + 1)
                if connection_modulators[to_idx][from_idx] is not None
            ],
            feature_combination_method='dense',
            require_definite_dimensions=['features'],
            connection_modulators=connection_modulators,
        )

    def get_dense_input(self, tensor_step: int, dense_tensor_list: list[torch.Tensor]):
        return torch.cat([
            self.connection_modulators[tensor_step][j](dense_tensor_list[j])
            for j in self.incoming_layer_connections[tensor_step]
        ], dim=-1)

    def forward(self, x, *args, **kwargs) -> torch.Tensor:
        dense_tensor_list: list[torch.Tensor] = [x]

        for i, layer in enumerate(self.layers):
            layer_dense_input = self.get_dense_input(i, dense_tensor_list)
            layer_out = layer(layer_dense_input, *args, **kwargs)
            dense_tensor_list.append(layer_out)

        return self.get_dense_input(self.num_layers, dense_tensor_list)

    @classmethod
    @override
    def from_layer_provider(
            cls,
            layer_provider: LayerProvider,
            modulator_provider: Callable[[int, int, int], Net | None],  # = lambda from_idx, to_idx, in_features: ...
            layers_sizes: list[int] = None,
            in_size: int = None,
            layer_out_sizes: list[int] = None,
            num_layers: int = None,
            num_features: int = None,
    ) -> 'ModulatedDenseSkipNet':

        layers_cum_in_out_sizes, connection_modulators = cls.compute_layers_cum_modulated_in_out_sizes(
                modulator_provider=modulator_provider,
                layers_sizes=layers_sizes,
                in_size=in_size,
                out_sizes=layer_out_sizes,
                num_layers=num_layers,
                num_features=num_features,
            )

        return ModulatedDenseSkipNet(
            layers=cls.provide_layers(layer_provider, layers_cum_in_out_sizes),
            connection_modulators=connection_modulators
        )

    @classmethod
    def compute_layers_cum_modulated_in_out_sizes(
            cls,
            modulator_provider: Callable[[int, int, int], Net | None],  # = lambda from_idx, to_idx, in_features: ...
            layers_sizes: list[int] = None,
            in_size: int = None,
            out_sizes: list[int] = None,
            num_layers: int = None,
            num_features: int = None,
    ):
        layers_in_out_sizes = SeqNet.resolve_sequential_in_out_features(
            layer_sizes=layers_sizes,
            in_size=in_size,
            out_sizes=out_sizes,
            num_layers=num_layers,
            num_features=num_features,
        )

        sizes = [layers_in_out_sizes[0][0]] + [out_size for in_size, out_size in layers_in_out_sizes]
        connection_modulators: list[list[Net | None]] = []
        layer_connections: list[tuple[int, int]] = []
        for to_idx in range(len(layers_in_out_sizes) + 1):
            layer_modulators = []
            for from_idx in range(to_idx + 1):
                modulator = modulator_provider(from_idx, to_idx, sizes[from_idx])
                layer_modulators.append(modulator)
                if modulator is not None:
                    layer_connections.append((from_idx, to_idx))
            connection_modulators.append(layer_modulators)

        num_layers = len(layers_in_out_sizes)
        layer_connections: np.ndarray = LayerConnections.to_np(layer_connections, num_layers)

        layers_cum_in_out_sizes: list[tuple[int, int]] = []
        for i in range(num_layers):
            in_size_sum = 0
            for incoming_tensor_step in cls.find_incoming_tensor_step_indices(i, layer_connections):
                incoming_tensor_step_size = sizes[incoming_tensor_step]
                in_size_sum += (connection_modulators[i][incoming_tensor_step]
                                .forward_shape(TensorShape(features=incoming_tensor_step_size))
                                .get_definite_size('features'))
            layers_cum_in_out_sizes.append((in_size_sum, layers_in_out_sizes[i][1]))

        return layers_cum_in_out_sizes, connection_modulators
