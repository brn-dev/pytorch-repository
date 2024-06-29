from typing import Callable

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
            initial_weight_calculator: Callable[[int, int], float] = None,

            return_dense: bool = False,
    ):
        super().__init__(
            layers=layers,
            layer_connections=layer_connections,
            feature_combination_method='additive',
            require_definite_dimensions=['features'],
        )
        self.num_features = self.in_shape.get_definite_size('features')
        self.return_dense = return_dense
        if return_dense:
            self.out_shape['features'] *= self.num_layers + 1

        weights = [torch.zeros((l + 1, self.num_features)) for l in range(self.num_layers + 1)]
        masks = [torch.zeros((l + 1, self.num_features)) for l in range(self.num_layers + 1)]

        for from_idx, to_idx in self.layer_connections:
            if initial_weight_calculator is not None:
                weights[to_idx][from_idx, :] = initial_weight_calculator(from_idx, to_idx)
            else:
                weights[to_idx][from_idx, :] = (initial_direct_connection_weight
                                                if from_idx == to_idx
                                                else initial_skip_connection_weight)
            masks[to_idx][from_idx, :] = 1.0

        self.weights: list[torch.Tensor] = []
        self.masks: list[torch.Tensor] = []
        for i, (weight, mask) in enumerate(zip(weights, masks)):
            weight_name = f'weights-{i}'
            self.register_parameter(weight_name, nn.Parameter(weight, requires_grad=weights_trainable))
            self.weights.append(self.__getattr__(weight_name))

            mask_name = f'mask-{i}'
            mask.requires_grad = False
            self.register_buffer(mask_name, mask)
            self.masks.append(self.__getattr__(weight_name))

    def forward(self, x: torch.Tensor, *args, **kwargs):
        dense_tensor = x.unsqueeze(-2)

        for i, layer in enumerate(self.layers):
            layer_input = (dense_tensor * self.masks[i] * self.weights[i]).sum(dim=-2)
            layer_output = layer(layer_input, *args, **kwargs)

            dense_tensor = torch.cat((dense_tensor, layer_output.unsqueeze(dim=-2)), dim=-2)

        if self.return_dense:
            return torch.flatten(dense_tensor, start_dim=-2, end_dim=-1)

        out = (dense_tensor * self.masks[-1] * self.weights[-1]).sum(dim=-2)
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


class FullyConnectedWeightedAdditiveSkipNet1(LayeredNet):

    def __init__(
            self,
            layers: NetListLike,

            weights_trainable: bool = False,

            initial_direct_connection_weight: float = 1.0,
            initial_skip_connection_weight: float = 1.0,
            initial_weight_calculator: Callable[[int, int], float] = None,

            return_dense: bool = False,
    ):
        super().__init__(
            layers=layers,
            layer_connections='full',
            feature_combination_method='additive',
            require_definite_dimensions=['features'],
        )
        self.num_features = self.in_shape.get_definite_size('features')
        self.return_dense = return_dense
        if return_dense:
            self.out_shape['features'] *= self.num_layers + 1

        weights = [torch.zeros((l + 1, self.num_features)) for l in range(self.num_layers + 1)]

        for from_idx, to_idx in self.layer_connections:
            if initial_weight_calculator is not None:
                weights[to_idx][from_idx, :] = initial_weight_calculator(from_idx, to_idx)
            else:
                weights[to_idx][from_idx, :] = (initial_direct_connection_weight
                                                if from_idx == to_idx
                                                else initial_skip_connection_weight)

        self.weights: list[torch.Tensor] = []
        for i, weight in enumerate(weights):
            weight_name = f'weights-{i}'
            self.register_parameter(weight_name, nn.Parameter(weight, requires_grad=weights_trainable))
            self.weights.append(self.__getattr__(weight_name))

    def forward(self, x: torch.Tensor, *args, **kwargs):
        dense_tensor = x.unsqueeze(-2)

        for i, layer in enumerate(self.layers):
            layer_input = (dense_tensor * self.weights[i]).sum(dim=-2)
            layer_output = layer(layer_input, *args, **kwargs)

            dense_tensor = torch.cat((dense_tensor, layer_output.unsqueeze(dim=-2)), dim=-2)

        if self.return_dense:
            return torch.flatten(dense_tensor, start_dim=-2, end_dim=-1)

        out = (dense_tensor * self.weights[-1]).sum(dim=-2)
        return out

    @classmethod
    @override
    def from_layer_provider(
            cls,
            layer_provider: LayerProvider,
            num_layers: int,
            num_features: int,

            weights_trainable: bool = False,
            initial_direct_connection_weight: float = 1.0,
            initial_skip_connection_weight: float = 1.0,

            return_dense: bool = False,
    ) -> 'FullyConnectedWeightedAdditiveSkipNet1':
        return FullyConnectedWeightedAdditiveSkipNet1(
            layers=cls.provide_layers(
                layer_provider=layer_provider,
                in_out_features=[(num_features, num_features) for _ in range(num_layers)]
            ),
            weights_trainable=weights_trainable,
            initial_direct_connection_weight=initial_direct_connection_weight,
            initial_skip_connection_weight=initial_skip_connection_weight,
            return_dense=return_dense,
        )

class FullyConnectedAdditiveSkipNet(LayeredNet):

    def __init__(
            self,
            layers: NetListLike,
    ):
        super().__init__(
            layers=layers,
            layer_connections='full',
            feature_combination_method='additive',
            require_definite_dimensions=['features'],
        )
        self.num_features = self.in_shape.get_definite_size('features')

    def forward(self, x: torch.Tensor, *args, **kwargs):
        summed_tensor = x

        for i, layer in enumerate(self.layers):
            layer_output = layer(summed_tensor, *args, **kwargs)

            summed_tensor = summed_tensor + layer_output

        return summed_tensor

    @classmethod
    @override
    def from_layer_provider(
            cls,
            layer_provider: LayerProvider,
            num_layers: int,
            num_features: int,
    ) -> 'FullyConnectedAdditiveSkipNet':
        return FullyConnectedAdditiveSkipNet(
            layers=cls.provide_layers(
                layer_provider=layer_provider,
                in_out_features=[(num_features, num_features) for _ in range(num_layers)]
            ),
        )
