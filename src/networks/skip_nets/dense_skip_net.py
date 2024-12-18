from overrides import override

import numpy as np
import torch

from src.networks.core.layer_connections import LayerConnections
from src.networks.core.layered_net import LayeredNet, LayerProvider
from src.networks.core.net import Net
from src.networks.core.net_list import NetListLike
from src.networks.core.seq_net import SeqNet


class DenseSkipNet(LayeredNet):

    def __init__(
            self,
            layers: NetListLike,
            layer_connections: LayerConnections.LayerConnectionsLike = 'full',
            feature_dim_index: int = -1,
    ):
        super().__init__(
            layers=layers,
            layer_connections=layer_connections,
            feature_combination_method='dense',
            require_definite_dimensions=['features'],
        )
        self.feature_dim_index = feature_dim_index

    def get_dense_input(self, tensor_step: int, dense_tensor_list: list[torch.Tensor]):
        return torch.cat([
            dense_tensor_list[j] for j in self.incoming_layer_connections[tensor_step]
        ], dim=self.feature_dim_index)

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
            layers_sizes: list[int] = None,
            in_size: int = None,
            out_sizes: list[int] = None,
            num_layers: int = None,
            num_features: int = None,
            layer_connections: LayerConnections.LayerConnectionsLike = 'full',
            feature_dim_index: int = -1,
    ) -> 'DenseSkipNet':
        return DenseSkipNet(
            layers=cls.provide_layers(layer_provider, cls.compute_layers_cum_in_out_sizes(
                layer_connections=layer_connections,
                layers_sizes=layers_sizes,
                in_size=in_size,
                out_sizes=out_sizes,
                num_layers=num_layers,
                num_features=num_features,
            )),
            layer_connections=layer_connections,
            feature_dim_index=feature_dim_index,
        )

    @classmethod
    def compute_layers_cum_in_out_sizes(
            cls,
            layer_connections: LayerConnections.LayerConnectionsLike,
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

        num_layers = len(layers_in_out_sizes)
        sizes = np.array([layers_in_out_sizes[0][0]] + [out_size for in_size, out_size in layers_in_out_sizes])
        layer_connections = LayerConnections.to_np(layer_connections, num_layers)

        layers_cum_in_out_sizes: list[tuple[int, int]] = []
        for i in range(num_layers):
            in_size_sum = sizes[cls.find_incoming_tensor_step_indices(i, layer_connections)].sum()
            layers_cum_in_out_sizes.append((in_size_sum, layers_in_out_sizes[i][-1]))

        return layers_cum_in_out_sizes


class FullyConnectedDenseSkipNet(DenseSkipNet):

    def __init__(
            self,
            layers: NetListLike,
            feature_dim_index: int = -1,
    ):
        super().__init__(
            layers=layers,
            layer_connections='full',
            feature_dim_index=feature_dim_index,
        )

    def forward(self, x, *args, **kwargs) -> torch.Tensor:
        dense_tensor = x

        for i, layer in enumerate(self.layers):
            layer_out = layer(dense_tensor, *args, **kwargs)
            dense_tensor = torch.cat((dense_tensor, layer_out), dim=self.feature_dim_index)

        return dense_tensor

    @classmethod
    def fully_connected_from_layer_provider(
            cls,
            layer_provider: LayerProvider,
            layers_sizes: list[int] = None,
            in_size: int = None,
            out_sizes: list[int] = None,
            num_layers: int = None,
            num_features: int = None,
            layer_connections: LayerConnections.LayerConnectionsLike = 'full',
            feature_dim_index: int = -1,
    ) -> 'FullyConnectedDenseSkipNet':
        return FullyConnectedDenseSkipNet(
            layers=cls.provide_layers(layer_provider, cls.compute_layers_cum_in_out_sizes(
                layer_connections=layer_connections,
                layers_sizes=layers_sizes,
                in_size=in_size,
                out_sizes=out_sizes,
                num_layers=num_layers,
                num_features=num_features,
            )),
            feature_dim_index=feature_dim_index,
        )
