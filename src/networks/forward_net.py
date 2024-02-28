from typing import Callable

from torch import nn

from src.networks.net_connections import NetConnections
from src.networks.nn_base import NNBase


class ForwardNet(NNBase):

    LayerProvider = Callable[[int, bool, int, int], nn.Module]

    @staticmethod
    def to_layers_in_out_sizes(
            layers_in_out_sizes: list[tuple[int, int]] = None,

            layers_sizes: list[int] = None,

            in_size: int = None,
            out_sizes: list[int] = None,

            num_layers: int = None,
            num_features: int = None
    ) -> list[tuple[int, int]]:
        parameter_choices = [layers_in_out_sizes, layers_sizes, in_size, num_layers]

        assert in_size is not None and out_sizes is not None or in_size is None and out_sizes is None
        assert parameter_choices.count(None) == len(parameter_choices) - 1, 'only one parameter choice can be used'

        if layers_sizes is not None:
            layers_in_out_sizes = [(layers_sizes[i], layers_sizes[i+1]) for i in range(len(layers_sizes) - 1)]
        if in_size is not None:
            layers_in_out_sizes = []
            for out_size in out_sizes:
                layers_in_out_sizes.append((in_size, out_size))
                in_size = out_size
        if num_layers is not None:
            layers_in_out_sizes = [(num_features, num_features) for _ in range(num_layers)]

        return layers_in_out_sizes

    @staticmethod
    def create_layers(
            layer_provider: LayerProvider,
            layers_in_out_sizes: list[tuple[int, int]] = None,
    ) -> nn.ModuleList:
        layers: list[nn.Module] = []

        for i, (in_size, out_size) in enumerate(layers_in_out_sizes):
            is_final_layer = i == len(layers_in_out_sizes) - 1
            layers.append(layer_provider(i, is_final_layer, in_size, out_size))

        return nn.ModuleList(layers)

    def __init__(
            self,
            layer_provider: LayerProvider,
            layers_in_out_sizes: list[tuple[int, int]] = None,
            layers_sizes: list[int] = None,
            in_size: int = None,
            out_sizes: list[int] = None,
            num_layers: int = None,
            num_features: int = None
    ):
        super().__init__()

        self.layers_in_out_sizes = self.to_layers_in_out_sizes(
            layers_in_out_sizes=layers_in_out_sizes,
            layers_sizes=layers_sizes,
            in_size=in_size,
            out_sizes=out_sizes,
            num_layers=num_layers,
            num_features=num_features
        )

        self.num_layers = len(self.layers_in_out_sizes)

        self.layers = nn.Sequential(*self.create_layers(layer_provider, self.layers_in_out_sizes))
        self.connections = NetConnections.by_name('sequential', self.num_layers)

    def forward(self, *args, **kwargs):
        self.layers(*args, **kwargs)


