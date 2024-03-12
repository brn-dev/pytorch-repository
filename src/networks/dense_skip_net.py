import numpy as np
import torch

from src.networks.core.seq_net import SeqNet
from src.networks.core.net import Net


class DenseSkipNet(SeqNet):

    @staticmethod
    def compute_layers_cum_in_out_sizes(
            layers_sizes: list[int] = None,
            in_size: int = None,
            out_sizes: list[int] = None,
            num_layers: int = None,
            num_features: int = None,
            connections: Net.LayerConnections.LayerConnectionsLike = 'full',
    ):
        layers_in_out_sizes = SeqNet.resolve_sequential_layer_in_out_sizes(
            layer_sizes=layers_sizes,
            in_size=in_size,
            out_sizes=out_sizes,
            num_layers=num_layers,
            num_features=num_features,
        )

        num_layers = len(layers_in_out_sizes)
        in_sizes = np.array([in_size for in_size, out_size in layers_in_out_sizes])
        connections = Net.LayerConnections.to_np(connections, num_layers)

        layers_cum_in_out_sizes: list[tuple[int, int]] = []
        for i in range(num_layers):
            in_size_sum = in_sizes[connections[connections[:, 1] == i][:, 0]].sum()
            layers_cum_in_out_sizes.append((in_size_sum, layers_in_out_sizes[i][-1]))

        return layers_cum_in_out_sizes

    def __init__(
            self,
            layer_provider: SeqNet.LayerProvider,
            layers_sizes: list[int] = None,
            in_size: int = None,
            out_sizes: list[int] = None,
            num_layers: int = None,
            num_features: int = None,
            connections: Net.LayerConnections.LayerConnectionsLike = 'full',
    ):
        super().__init__(
            layer_provider,
            layers_in_out_sizes=DenseSkipNet.compute_layers_cum_in_out_sizes(
                layers_sizes=layers_sizes,
                in_size=in_size,
                out_sizes=out_sizes,
                num_layers=num_layers,
                num_features=num_features,
                connections=connections
            )
        )
        self.connections = self.LayerConnections.to_np(connections, self.num_layers)


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
