import torch

from src.networks.net import Net, LayeredNet, LayerList


class ParallelNet(LayeredNet):

    @staticmethod
    def from_layer_provider(
            layer_provider: Net.LayerProvider,
            in_size: int = None,
            out_sizes: list[int] = None,
            num_layers: int = None,
            num_features: int = None
    ) -> 'ParallelNet':
        assert in_size is None and out_sizes is None or in_size is not None and out_sizes is not None
        assert [in_size, num_layers].count(None) == 1

        if num_layers is not None:
            in_size = num_features
            out_sizes = [num_features] * num_layers

        return ParallelNet(ParallelNet.create_layer_list(
            layer_provider,
            [(in_size, out_size) for out_size in out_sizes]
        ))

    def __init__(self, layers: LayerList):
        super().__init__(
            in_features=layers[0].in_features,
            out_features=sum(layer.out_features for layer in layers),
            layers=layers,
            layer_connections=LayeredNet.LayerConnections.by_name('parallel', len(layers))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs: list[torch.Tensor] = []

        for layer in self._layers:
            outs.append(layer(x))

        return torch.cat(outs, dim=-1)
