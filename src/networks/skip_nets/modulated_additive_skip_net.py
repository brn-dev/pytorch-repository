from typing import Callable

import torch
from overrides import override

from src.networks.core.layered_net import LayeredNet, LayerProvider
from src.networks.core.net import Net
from src.networks.core.net_list import NetListLike


class ModulatedAdditiveSkipNet(LayeredNet):

    def __init__(
            self,
            layers: NetListLike,
            connection_modulators: list[list[Net | None]],
            return_dense: bool = False,
    ):
        super().__init__(
            layers=layers,
            layer_connections=[
                (from_idx, to_idx)
                for to_idx in range(len(layers) + 1)
                for from_idx in range(to_idx + 1)
                if connection_modulators[to_idx][from_idx] is not None
            ],
            combination_method='additive',
            require_definite_dimensions=['features'],
            connection_modulators=connection_modulators,
        )
        self.num_features = self.in_shape.get_definite_size('features')
        self.return_dense = return_dense
        if return_dense:
            self.out_shape['features'] *= self.num_layers + 1

    def get_additive_input(self, tensor_layer: int, dense_tensor_list: list[torch.Tensor]):
        return torch.stack([
            self.connection_modulators[tensor_layer][j](dense_tensor_list[j])
            for j
            in self.incoming_layer_connections[tensor_layer]
        ], dim=-2).sum(dim=-2)

    def forward(self, x, *args, **kwargs) -> torch.Tensor:
        dense_tensor_list: list[torch.Tensor] = [x]

        for i, layer in enumerate(self.layers):
            layer_dense_input = self.get_additive_input(i, dense_tensor_list)
            layer_out = layer(layer_dense_input, *args, **kwargs)
            dense_tensor_list.append(layer_out)

        if self.return_dense:
            return torch.cat(dense_tensor_list, dim=-1)
        else:
            return self.get_additive_input(self.num_layers, dense_tensor_list)

    @classmethod
    @override
    def from_layer_provider(
            cls,
            layer_provider: LayerProvider,
            modulator_provider: Callable[[int, int, int], Net | None],  # = lambda from_idx, to_idx, num_features: ...
            num_layers: int = None,
            num_features: int = None,
            return_dense: bool = False,
    ) -> 'ModulatedAdditiveSkipNet':
        return ModulatedAdditiveSkipNet(
            layers=cls.provide_layers(layer_provider, [
                (num_features, num_features)
                for _ in range(num_layers)
            ]),
            connection_modulators=[
                [
                    modulator_provider(from_idx, to_idx, num_features)
                    for from_idx in range(to_idx + 1)
                ]
                for to_idx in range(num_layers + 1)
            ],
            return_dense=return_dense
        )

