from typing import Callable

from torch import nn

from src.networks.nn_base import NNBase


class LayerStack(NNBase):

    LayerProvider = Callable[[int, bool, int, int], nn.Module]

    @staticmethod
    def compute_layers_in_out_sizes(
            layers_in_out_sizes: list[tuple[int, int]] = None,
            layers_sizes: list[int] = None,
            num_layers: int = None,
            num_features: int = None
    ):
        # TODO
        # assert num_layers is not None and num_features is not None or num_layers is None and num_features is None

        assert layers_in_out_sizes is None or layers_sizes is None
        assert layers_in_out_sizes is None or num_layers is None
        assert layers_sizes is None or num_layers is None

        assert layers_in_out_sizes is not None or layers_sizes is not None or num_layers is not None

        if layers_sizes is not None:
            layers_in_out_sizes = [(layers_sizes[i], layers_sizes[i+1]) for i in range(len(layers_sizes) - 1)]
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
            num_layers: int = None,
            num_features: int = None,
    ):
        super().__init__()

        self.layers_in_out_sizes = self.compute_layers_in_out_sizes(
            layers_in_out_sizes,
            layers_sizes,
            num_layers,
            num_features
        )

        self.num_layers = len(self.layers_in_out_sizes)

        self.layers = nn.Sequential(*self.create_layers(layer_provider, self.layers_in_out_sizes))

    def forward(self, *args, **kwargs):
        self.layers(*args, **kwargs)


