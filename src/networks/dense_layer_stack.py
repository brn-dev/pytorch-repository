import abc
from typing import Literal

import numpy as np
import torch
from torch import nn

from src.networks.layer_stack import LayerStack
from src.networks.weighing import WeighingTrainableChoices

class DenseLayerStack(LayerStack, abc.ABC):

    ConnectionsType = tuple[tuple[int, int]] | list[list[int]] | np.ndarray \
                      | Literal['dense', 'sequential']  # TODO Module

    @staticmethod
    def connections_to_np(
            connections: ConnectionsType,
            num_layers: int,
    ) -> np.ndarray:
        if isinstance(connections, str):
            if connections in ['dense', 'sequential']:
                # noinspection PyTypeChecker
                connections = DenseLayerStack.create_connections_by_name(connections, num_layers)
            else:
                raise ValueError('Unknown connection type name')
        else:
            connections = np.array(connections)

        connections = (connections % (num_layers + 1))
        return connections.astype(int)

    @staticmethod
    def create_connections_by_name(
            name: Literal['dense', 'sequential'],
            num_layers: int,
    ) -> np.ndarray:
        if name == 'dense':
            connections = np.array([
                [i, j]
                for i in range(0, num_layers + 1)
                for j in range(i, num_layers + 1)
            ])
        elif name == 'sequential':
            connections = np.array([
                [i, i]
                for i in range(0, num_layers + 1)
            ])
        else:
            raise ValueError

        connections = connections
        return connections


class AdditiveDenseLayerStack(DenseLayerStack):

    def __init__(
            self,
            layer_provider: LayerStack.LayerProvider,
            num_layers: int,
            num_features: int = None,

            connections: DenseLayerStack.ConnectionsType = 'dense',

            weights_trainable: WeighingTrainableChoices = False,
            initial_direct_connection_weight: float = 1.0,
            initial_skip_connection_weight: float = 1.0,

            # dropout_p: float = 0.0,
            # normalization_provider: NNBase.Provider = None,
    ):
        super().__init__(layer_provider, num_layers=num_layers, num_features=num_features)

        connections = self.connections_to_np(connections, num_layers)

        assert (connections >= 0).all()
        assert (connections <= num_layers).all()
        assert (connections[:, 0] <= connections[:, 1]).all()
        assert len(np.unique(connections[:, 0])) == num_layers + 1, np.unique(connections[:, 0])
        assert len(np.unique(connections[:, 1])) == num_layers + 1, np.unique(connections[:, 1])

        mask = torch.zeros((num_layers + 1, num_features, num_layers + 1))
        weight = torch.zeros((num_layers + 1, num_features, num_layers + 1))

        for from_idx, to_idx in connections:
            mask[to_idx, :, from_idx] = 1.0
            weight[to_idx, :, from_idx] = (initial_direct_connection_weight
                                           if from_idx == to_idx
                                           else initial_skip_connection_weight)

        self.connections = nn.Parameter(torch.FloatTensor(connections), requires_grad=False)
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.weight = nn.Parameter(weight, requires_grad=weights_trainable)

    def forward(self, x: torch.Tensor, return_dense=False, **kwargs):

        dense_tensor = torch.zeros_like(x.float()) \
                .unsqueeze(-1).repeat_interleave(self.num_layers + 1, dim=-1)
        dense_tensor[..., 0] = x

        for i, layer in enumerate(self.layers):
            layer_input = (dense_tensor * self.mask[i] * self.weight[i]).sum(dim=-1)
            layer_output = layer(layer_input, **kwargs)
            dense_tensor[..., i + 1] = layer_output

        out = (dense_tensor * self.mask[-1] * self.weight[-1]).sum(dim=-1)

        if return_dense:
            return out, dense_tensor
        else:
            return out


class ConcatDenseLayerStack(DenseLayerStack):

    @staticmethod
    def compute_layers_cum_in_out_sizes(
            layers_in_out_sizes: list[tuple[int, int]] = None,
            layers_sizes: list[int] = None,
            num_layers: int = None,
            num_features: int = None,
            connections: DenseLayerStack.ConnectionsType = 'dense',
    ):
        layers_in_out_sizes = DenseLayerStack.compute_layers_in_out_sizes(
            layers_in_out_sizes, layers_sizes, num_layers, num_features)

        num_layers = len(layers_in_out_sizes)
        in_sizes = np.array([in_size for in_size, out_size in layers_in_out_sizes])
        connections = DenseLayerStack.connections_to_np(connections, num_layers)

        layers_cum_in_out_sizes: list[tuple[int, int]] = []
        for i in range(num_layers):
            in_size_sum = in_sizes[connections[connections[:, 1] == i][:, 0]].sum()
            layers_cum_in_out_sizes.append((in_size_sum, layers_in_out_sizes[i][-1]))

        return layers_cum_in_out_sizes

    def __init__(
            self,
            layer_provider: LayerStack.LayerProvider,
            num_layers: int = None,
            num_features: int = None,
            layers_sizes: list[int] = None,
            connections: DenseLayerStack.ConnectionsType = 'dense',
    ):
        super().__init__(
            layer_provider,
            layers_in_out_sizes=ConcatDenseLayerStack.compute_layers_cum_in_out_sizes(
                None, layers_sizes=layers_sizes,
                num_layers=num_layers, num_features=num_features,
                connections=connections
            )
        )
        self.connections = DenseLayerStack.connections_to_np(connections, self.num_layers)


    def forward(self, x, *args, return_dense=False, return_dense_list=False, **kwargs):
        layer_out = None
        dense_tensor_list: list[torch.Tensor] = [x]

        for i, layer in enumerate(self.layers):
            sources = self.connections[self.connections[:, 1] == i][:, 0]
            layer_dense_input = torch.cat([
                dense_source
                for j, dense_source in enumerate(dense_tensor_list)
                if j in sources
            ])
            layer_out = layer(layer_dense_input)
            dense_tensor_list.append(layer_out)

        if return_dense:
            return layer_out, torch.cat(dense_tensor_list)
        if return_dense_list:
            return layer_out, dense_tensor_list
        return layer_out
