import abc
from typing import Literal, Callable, TypeVar

import numpy as np
from torch import nn

from src.networks.net import Net
from src.networks.net_list import NetList, NetListLike


class LayeredNet(Net, abc.ABC):

    class LayerConnections:

        Presets = Literal['full', 'sequential', 'parallel']
        LayerConnectionsLike = list[list[int, int]] | np.ndarray | Presets

        @staticmethod
        def by_name(
                name: Presets | str,
                num_layers: int,
        ) -> np.ndarray:
            if name == 'full':
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

        if not layer.in_out_features_defined():
            raise ValueError(f'Provider returned layer with undefined in/out features! ', layer)

        return layer

    @classmethod
    def create_layer_list(
            cls,
            layer_provider: LayerProvider,
            layers_in_out_sizes: list[tuple[int, int]] = None,
    ) -> NetList:
        layers: list[Net] = []

        for i, (in_size, out_size) in enumerate(layers_in_out_sizes):
            is_final_layer = i == len(layers_in_out_sizes) - 1

            layer = cls.provide_layer(layer_provider, i, is_final_layer, in_size, out_size)
            layers.append(layer)

        return NetList(layers)

    @staticmethod
    @abc.abstractmethod
    def from_layer_provider(layer_provider: LayerProvider, *args, **kwargs) -> 'LayeredNetDerived':
        raise NotImplemented

    def __init__(
            self,
            in_features: Net.InFeaturesType,
            out_features: Net.OutFeaturesType,
            layers: NetListLike,
            layer_connections: LayerConnections.LayerConnectionsLike,
            allow_undefined_in_out_features: bool = False,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            allow_undefined_in_out_features=allow_undefined_in_out_features,
        )

        self.layers = NetList.as_net_list(layers)
        self.layer_connections = layer_connections


LayeredNetDerived = TypeVar('LayeredNetDerived', bound=LayeredNet)
