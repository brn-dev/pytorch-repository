from overrides import override

import torch
from torch import nn

from src.networks.core.layer_connections import LayerConnections
from src.networks.core.layered_net import LayeredNet, LayerProvider
from src.networks.core.net_list import NetListLike


class AdditiveSkipNet(LayeredNet):

    def __init__(
            self,
            layers: NetListLike,

            layer_connections: LayerConnections.LayerConnectionsLike = 'full',

            weights_trainable: bool = False,
            initial_direct_connection_weight: float = 1.0,
            initial_skip_connection_weight: float = 1.0,

            return_dense: bool = False,
    ):
        super().__init__(
            layers=layers,
            layer_connections=layer_connections,
            combination_method='additive',
            require_definite_dimensions=['features'],
        )
        self.num_features = self.in_shape.get_definite_size('features')
        self.return_dense = return_dense
        if return_dense:
            self.out_shape['features'] *= self.num_layers + 1

        mask = torch.zeros((self.num_layers + 1, self.num_layers + 1, self.num_features))
        weight = torch.zeros((self.num_layers + 1, self.num_layers + 1, self.num_features))

        for from_idx, to_idx in layer_connections:
            mask[to_idx, from_idx, :] = 1.0
            weight[to_idx, from_idx, :] = (initial_direct_connection_weight
                                           if from_idx == to_idx
                                           else initial_skip_connection_weight)

        self.mask = nn.Parameter(mask, requires_grad=False)
        self.weight = nn.Parameter(weight, requires_grad=weights_trainable)

    def forward(self, x: torch.Tensor, *args, **kwargs):

        dense_tensor = torch.zeros_like(x.float()) \
                .unsqueeze(-1).repeat_interleave(self.num_layers + 1, dim=-2)
        dense_tensor[..., 0, :] = x

        for i, layer in enumerate(self.layers):
            layer_input = (
                    dense_tensor[..., :(i+1), :]
                    * self.mask[i, :(i+1), :]
                    * self.weight[i, :(i+1), :]
            ).sum(dim=-2)
            layer_output = layer(layer_input, *args, **kwargs)
            dense_tensor[..., i+1, :] = layer_output

        if self.return_dense:
            return torch.flatten(dense_tensor, start_dim=-2, end_dim=-1)

        out = (dense_tensor * self.mask[-1] * self.weight[-1]).sum(dim=-2)
        return out

    @classmethod
    @override
    def from_layer_provider(
            cls,
            layer_provider: LayerProvider,
            num_layers: int,
            num_features: int,

            layer_connections: LayerConnections.LayerConnectionsLike = 'full',

            weights_trainable: bool = False,
            initial_direct_connection_weight: float = 1.0,
            initial_skip_connection_weight: float = 1.0,

            return_dense: bool = False,
    ) -> 'AdditiveSkipNet':
        return AdditiveSkipNet(
            layers=cls.provide_layers(
                layer_provider=layer_provider,
                in_out_features=[(num_features, num_features) for _ in range(num_layers)]
            ),
            layer_connections=layer_connections,
            weights_trainable=weights_trainable,
            initial_direct_connection_weight=initial_direct_connection_weight,
            initial_skip_connection_weight=initial_skip_connection_weight,
            return_dense=return_dense,
        )
