from typing import Callable

from torch import nn

from src.networks.nn_base import NNBase


class LayerStack(NNBase):

    def __init__(
            self,
            layer_provider: Callable[[int, int, bool], nn.Module],
            layers_in_out_sizes: list[tuple] = None,
            layers_sizes: list[int] = None,
            num_layers: int = None,
            num_features: int = None
    ):
        super().__init__()

        assert num_layers is not None and num_features is not None or num_layers is None and num_features is None

        assert layers_in_out_sizes is None or layers_sizes is None
        assert layers_in_out_sizes is None or num_layers is None
        assert layers_sizes is None or num_layers is None

        assert layers_in_out_sizes is not None or layers_sizes is not None or num_layers is not None

        if layers_sizes is not None:
            layers_in_out_sizes = [(layers_sizes[i], layers_sizes[i+1]) for i in range(len(layers_sizes) - 1)]
        if num_layers is not None:
            layers_in_out_sizes = [(num_features, num_features) for _ in range(num_layers)]


        self.layers = self.create_layers(layer_provider, layers_in_out_sizes)
        self.layer_stack = self.combine_layers(self.layers)

    @staticmethod
    def create_layers(
            layer_provider: Callable[[int, int, bool], nn.Module],
            layers_in_out_sizes: list[tuple] = None,
    ) -> list[nn.Module]:
        layers: list[nn.Module] = []

        for i, (in_size, out_size) in enumerate(layers_in_out_sizes):
            is_final_layer = i == len(layers_in_out_sizes) - 1
            layers.append(layer_provider(in_size, out_size, is_final_layer))

        return layers

    @staticmethod
    def combine_layers(layers: list[nn.Module]) -> nn.Module:
        return nn.Sequential(*layers)

    def forward(self, *args, **kwargs):
        self.layer_stack(*args, **kwargs)


