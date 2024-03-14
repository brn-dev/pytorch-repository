from overrides import override

import numpy as np

from src.networks.core.layer_connections import LayerConnections
from src.networks.core.layered_net import LayeredNet, LayerProvider, ShapeCombinationMethod
from src.networks.core.net_list import NetList, NetListLike
from src.networks.core.seq_shape import find_seq_in_out_shapes
from src.networks.core.tensor_shape import TensorShape
from src.utils import all_none_or_all_not_none, one_not_none


class SeqNet(LayeredNet):

    def __init__(self, layers: NetListLike):
        super().__init__(
            layers=layers,
            layer_connections=LayerConnections.by_name('sequential', len(layers)),
            combination_method=None,
        )

    @staticmethod
    @override
    def find_in_out_shapes(
            layers: NetList,
            layer_connections: np.ndarray,
            combination_method: ShapeCombinationMethod
    ) -> tuple[TensorShape, TensorShape]:
        in_shape, out_shape = find_seq_in_out_shapes(layers)
        return in_shape, out_shape

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


    @staticmethod
    def resolve_sequential_in_out_features(
            layer_sizes: list[int] = None,

            in_size: int = None,
            out_sizes: list[int] = None,

            num_layers: int = None,
            num_features: int = None
    ) -> list[tuple[int, int]]:
        parameter_choices = [layer_sizes, in_size, num_layers]

        assert all_none_or_all_not_none(in_size, out_sizes)
        assert one_not_none(parameter_choices), 'only one parameter choice must be used'

        if layer_sizes is not None:
            in_out_features = [(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

        elif in_size is not None:
            in_out_features = []
            for out_size in out_sizes:
                in_out_features.append((in_size, out_size))
                in_size = out_size

        elif num_layers is not None:
            in_out_features = [(num_features, num_features) for _ in range(num_layers)]

        else:
            raise Exception('This should not happen')

        return in_out_features


    @classmethod
    @override
    def from_layer_provider(
            cls,
            layer_provider: LayerProvider,
            in_out_features: list[tuple[int, int]] = None,
            layers_sizes: list[int] = None,
            in_size: int = None,
            out_sizes: list[int] = None,
            num_layers: int = None,
            num_features: int = None
    ) -> 'SeqNet':
        if in_out_features is None:
            in_out_features = cls.resolve_sequential_in_out_features(
                layer_sizes=layers_sizes,
                in_size=in_size,
                out_sizes=out_sizes,
                num_layers=num_layers,
                num_features=num_features
            )

        layers = cls.provide_layers(layer_provider, in_out_features)
        return SeqNet(layers)
