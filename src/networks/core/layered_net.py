import abc
from typing import Callable, TypeVar

import numpy as np
from torch import nn

from src.networks.core.layer_connections import LayerConnections
from src.networks.core.net import Net
from src.networks.core.net_list import NetList, NetListLike
from src.networks.core.tensor_shape import TensorShape

LayerProvider = Callable[[int, bool, int, int], Net | nn.Module]


class LayeredNet(Net, abc.ABC):

    def __init__(
            self,
            in_shape: TensorShape,
            out_shape: TensorShape,
            layers: NetListLike,
            layer_connections: LayerConnections.LayerConnectionsLike,
    ):
        Net.__init__(
            self,
            in_shape=in_shape,
            out_shape=out_shape,
        )
        self.layers = NetList.as_net_list(layers)
        self.layer_connections: np.ndarray = LayerConnections.to_np(layer_connections, len(self.layers))

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
        layer = cls.as_net(layer)
        return layer

    @classmethod
    def provide_layers(
            cls,
            layer_provider: LayerProvider,
            in_out_features: list[tuple[int, int]] = None,
    ) -> NetList:
        layers: list[Net] = []

        for layer_nr, (in_features, out_features) in enumerate(in_out_features):
            is_final_layer = layer_nr == len(in_out_features) - 1

            layer = cls.provide_layer(layer_provider, layer_nr, is_final_layer, in_features, out_features)
            layers.append(layer)

        return NetList(layers)

    @staticmethod
    @abc.abstractmethod
    def from_layer_provider(layer_provider: LayerProvider, *args, **kwargs) -> 'LayeredNetDerived':
        raise NotImplemented


LayeredNetDerived = TypeVar('LayeredNetDerived', bound=LayeredNet)
