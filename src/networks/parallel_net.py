import torch

from src.networks.core.layer_connections import LayerConnections
from src.networks.core.layered_net import LayeredNet, LayerProvider
from src.networks.core.net_list import NetList


class ParallelNet(LayeredNet):

    def __init__(self, layers: NetList):
        first_layer_in_shape = layers[0].in_shape
        first_layer_out_shape = layers[0].out_shape
        out_features_sum = 0

        for i, layer in enumerate(layers):
            layer_in_features_definite, layer_in_features_size = layer.in_shape.try_get_definite_size('features')
            layer_out_features_definite, layer_out_features_size = layer.out_shape.try_get_definite_size('features')

            if not layer_in_features_definite:
                raise ValueError(f'In features of layer {i} ({layer}) are not definite')
            if not layer_out_features_definite:
                raise ValueError(f'In features of layer {i} ({layer}) are not definite')
            if not layer.accepts_shape(first_layer_in_shape):
                raise ValueError(f'Layer {i} ({layer}) does not accept the same '
                                 f'shape as layer 1 ({first_layer_in_shape})')

            for dim in layer.out_shape.dimension_names:
                if dim != 'features' and layer.out_shape[dim] != first_layer_out_shape[dim]:
                    raise ValueError(f'Layer {i} ({layer}) outputs a different size ({layer.out_shape}) in '
                                     f'dimension {dim} than the first layer {first_layer_out_shape}')


            out_features_sum += layer_out_features_size

        in_shape = first_layer_in_shape

        out_shape = first_layer_out_shape.copy()
        out_shape['features'] = out_features_sum

        super().__init__(
            in_shape=in_shape,
            out_shape=out_shape,
            layers=layers,
            layer_connections=LayerConnections.by_name('parallel', len(layers))
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
