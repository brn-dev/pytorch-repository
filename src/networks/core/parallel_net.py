import torch

from src.networks.core.layer_connections import LayerConnections
from src.networks.core.layered_net import LayeredNet, LayerProvider
from src.networks.core.net_list import NetList
from src.networks.core.tensor_shape import TensorShapeError


class ParallelNet(LayeredNet):

    def __init__(self, layers: NetList):
        super().__init__(
            layers=layers,
            layer_connections=LayerConnections.by_name('parallel', len(layers)),
            combination_method='dense',
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs: list[torch.Tensor] = []

        for layer in self.layers:
            outs.append(layer(x))

        return torch.cat(outs, dim=-1)


    @staticmethod
    def resolve_parallel_in_out_features(
            in_size: int = None,
            out_sizes: list[int] = None,

            num_layers: int = None,
            num_features: int = None
    ) -> list[tuple[int, int]]:
        assert in_size is None and out_sizes is None or in_size is not None and out_sizes is not None
        assert [in_size, num_layers].count(None) == 1

        if num_layers is not None:
            in_size = num_features
            out_sizes = [num_features] * num_layers

        in_out_features = [(in_size, out_size) for out_size in out_sizes]
        return in_out_features


    @staticmethod
    def from_layer_provider(
            layer_provider: LayerProvider,
            in_size: int = None,
            out_sizes: list[int] = None,
            num_layers: int = None,
            num_features: int = None
    ) -> 'ParallelNet':

        in_out_features = ParallelNet.resolve_parallel_in_out_features(
            in_size, out_sizes, num_layers, num_features
        )

        layers = ParallelNet.provide_layers(layer_provider, in_out_features)
        return ParallelNet(layers)
