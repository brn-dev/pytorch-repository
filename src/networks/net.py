import abc
from typing import Literal, Callable, Union, TypeVar, Iterator

import numpy as np
import torch
from torch import nn

from src.torch_nn_modules import is_nn_activation_module, is_nn_dropout_module, is_nn_pooling_module, \
    is_nn_padding_module, is_instance_of_group, is_nn_convolutional_module, \
    is_nn_linear_module, is_nn_identity_module


class Net(nn.Module, abc.ABC):

    IN_FEATURES_ANY = 'any'
    OUT_FEATURES_SAME = 'same'

    def __init__(self, in_features: int | Literal['any'], out_features: int | Literal['same']):
        super().__init__()
        self.in_features = in_features.lower() if isinstance(in_features, str) else in_features
        self.out_features = out_features.lower() if isinstance(out_features, str) else out_features

    def actualize_out_features(self, in_features: int) -> int:
        if self.in_features != Net.IN_FEATURES_ANY and self.in_features != in_features:
            raise ValueError('Argument in_features must be same as attribute self.in_features '
                             'if self.in_features is not equal to "any"')
        if self.out_features == Net.OUT_FEATURES_SAME:
            return in_features
        return self.out_features

    @staticmethod
    def as_net(module: Union['Net', nn.Module]) -> 'Net':
        if isinstance(module, Net):
            return module
        return TorchModuleWrapper(module)


class TorchModuleWrapper(Net):

    def __init__(self, module: nn.Module):
        if (is_nn_activation_module(module) or is_nn_dropout_module(module)
                or is_nn_pooling_module(module) or is_nn_padding_module(module)
                or is_nn_identity_module(module)):
            in_features, out_features = Net.IN_FEATURES_ANY, Net.OUT_FEATURES_SAME

        elif is_instance_of_group(module, [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                           nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]):
            in_features, out_features = module.num_features, module.num_features

        elif isinstance(module, nn.LayerNorm):
            num_features = module.normalized_shape
            if is_instance_of_group(num_features, [list, tuple, torch.Size]):
                num_features = num_features[-1]
            in_features, out_features = num_features, num_features

        elif is_nn_convolutional_module(module):
            in_features, out_features = module.in_channels, module.out_channels

        elif is_nn_linear_module(module):
            in_features, out_features = module.in_features, module.out_features

        elif isinstance(module, nn.Sequential):
            in_features, out_features = Net.IN_FEATURES_ANY, Net.OUT_FEATURES_SAME

            for sub_module in module:
                sub_module = TorchModuleWrapper(sub_module)
                if in_features == Net.IN_FEATURES_ANY and sub_module.in_features != Net.IN_FEATURES_ANY:
                    in_features = sub_module.in_features
                if sub_module.out_features != Net.OUT_FEATURES_SAME:
                    out_features = sub_module.out_features

        else:
            raise ValueError(f'Unknown module type {type(module)}')

        super().__init__(in_features, out_features)
        self.torch_module = module

    def forward(self, *args, **kwargs):
        self.torch_module(*args, **kwargs)


class LayerList(nn.ModuleList):

    def __init__(self, layers: list[Union[Net]]):
        super().__init__(modules=layers)
        self.layers = layers

    def __getitem__(self, idx: int | slice) -> Union[Net, 'LayerList']:
        if isinstance(idx, slice):
            return self.__class__(self.layers[idx])
        else:
            return self.layers[idx]

    def __iter__(self) -> Iterator[Union[Net]]:
        return iter(self.layers)


class LayeredNet(Net, abc.ABC):

    class LayerConnections:

        Presets = Literal['dense', 'sequential', 'parallel']
        LayerConnectionsLike = list[list[int, int]] | np.ndarray | Presets

        @staticmethod
        def by_name(
                name: Presets | str,
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
            elif name == 'parallel':
                connections = np.array(
                    [[0, i] for i in range(0, num_layers)]
                    + [[i, num_layers] for i in range(1, num_layers + 1)]
                )
            else:
                raise ValueError('Unknown connections name')

            return connections.astype(int)

        @staticmethod
        def to_np(layer_connections_like: LayerConnectionsLike, num_layers: int) -> np.ndarray:
            if isinstance(layer_connections_like, str):
                connections = LayeredNet.LayerConnections.by_name(layer_connections_like, num_layers)
            else:
                connections = np.array(layer_connections_like).astype(int)
                connections = (connections % (num_layers + 1))

                assert LayeredNet.LayerConnections.is_valid(connections, num_layers)

            return connections

        @staticmethod
        def is_valid(connections: np.ndarray, num_layers: int) -> bool:
            return ((connections >= 0).all()
                    and (connections <= num_layers).all()
                    and len(np.unique(connections[:, 0])) == num_layers + 1
                    and len(np.unique(connections[:, 1])) == num_layers + 1
                    and (connections[:, 0] <= connections[:, 1]).all()
                    )

    LayerProvider = Callable[[int, bool, int, int], Net | nn.Module]

    @classmethod
    def provide_layer(
            cls,
            provider: LayerProvider,
            layer_nr: int,
            is_last_layer: bool,
            in_features: int,
            out_features: int
    ) -> Net:
        layer = provider(layer_nr, is_last_layer, in_features, out_features)
        layer = Net.as_net(layer)

        if layer.in_features == Net.IN_FEATURES_ANY or layer.out_features == Net.OUT_FEATURES_SAME:
            raise ValueError(f'Provider returned layer with undefined in/out features! '
                             f'({layer.in_features = }, {layer.out_features = })', layer)

        return layer

    @classmethod
    def create_layer_list(
            cls,
            layer_provider: LayerProvider,
            layers_in_out_sizes: list[tuple[int, int]] = None,
    ) -> LayerList:
        layers: list[Net] = []

        for i, (in_size, out_size) in enumerate(layers_in_out_sizes):
            is_final_layer = i == len(layers_in_out_sizes) - 1

            layer = cls.provide_layer(layer_provider, i, is_final_layer, in_size, out_size)
            layers.append(layer)

        return LayerList(layers)

    @staticmethod
    @abc.abstractmethod
    def from_layer_provider(layer_provider: LayerProvider, *args, **kwargs) -> 'LayeredNetDerived':
        raise NotImplemented

    def __init__(
            self,
            in_features: int | Literal['any'],
            out_features: int | Literal['same'],
            layers: LayerList,
            layer_connections: LayerConnections.LayerConnectionsLike
    ):
        super().__init__(in_features, out_features)
        self.layers = layers
        self.layer_connections = layer_connections


LayeredNetDerived = TypeVar('LayeredNetDerived', bound=LayeredNet)
