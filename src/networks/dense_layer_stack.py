from typing import Callable, Literal

import numpy as np
import torch
from torch import nn

from src.networks.nn_base import NNBase
from src.networks.skip_connection import AdditiveSkipConnection, ConcatSkipConnection
from src.networks.layer_stack import LayerStack
from src.networks.weighing import WeighingTypes, WeighingTrainableChoices


class AdditiveDenseLayerStack(LayerStack):

    def __init__(
            self,
            layer_provider: Callable[[int, int, bool], nn.Module],
            num_layers: int = None,
            num_features: int = None,

            connections: tuple[tuple[int]] | list[list[int]] | np.ndarray |  # TODO: module
                        Literal['dense', 'sequential'] = 'dense',

            weights_trainable: WeighingTrainableChoices = False,
            initial_direct_connection_weight: WeighingTypes = 1.0,
            initial_skip_connection_weight: WeighingTypes = 1.0,

            # dropout_p: float = 0.0,
            # normalization_provider: NNBase.Provider = None,
    ):
        super().__init__(layer_provider, num_layers=num_layers, num_features=num_features)

        if isinstance(connections, str):
            if connections in ['dense', 'sequential']:
                # noinspection PyTypeChecker
                connections = self.create_connections_by_name(connections, num_layers)
            else:
                raise ValueError('Unknown connection type name')
        else:
            connections = np.array(connections)

        connections = (connections % (num_layers + 1))

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

        self.num_layers = num_layers
        self.num_features = num_features
        self.connections = connections

        self.mask = nn.Parameter(mask, requires_grad=False)
        self.weight = nn.Parameter(weight, requires_grad=weights_trainable)


    def forward(self, x: torch.Tensor, return_connection_node_values=False, **kwargs):
        connection_node_values = torch.zeros_like(x.float()) \
                .unsqueeze(-1).repeat_interleave(self.num_layers + 1, dim=-1)
        connection_node_values[..., 0] = x

        for i, layer in enumerate(self.layers):

            layer_input = (connection_node_values * self.mask[i] * self.weight[i]).sum(dim=-1)
            layer_output = layer(layer_input, **kwargs)
            connection_node_values[..., i + 1] = layer_output

        if return_connection_node_values:
            return connection_node_values
        else:
            return connection_node_values[..., -1]

    @staticmethod
    def create_connections_by_name(
            name: Literal['dense', 'sequential'],
            num_layers: int,
    ):
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

        connections: np.ndarray = (connections % (num_layers + 1))
        return connections





