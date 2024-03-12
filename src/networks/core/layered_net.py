import abc
from typing import Callable, TypeVar, Literal, Iterable

import numpy as np
from torch import nn

from src.networks.core.layer_connections import LayerConnections
from src.networks.core.net import Net
from src.networks.core.net_list import NetList, NetListLike
from src.networks.core.tensor_shape import TensorShape, TensorShapeError

LayerProvider = Callable[[int, bool, int, int], Net | nn.Module]

ShapeCombinationMethod = Literal['additive', 'dense', None]


class LayeredNet(Net, abc.ABC):

    def __init__(
            self,
            layers: NetListLike,
            layer_connections: LayerConnections.LayerConnectionsLike,
            combination_method: ShapeCombinationMethod,
            definite_dimensions: Iterable[str] = (),
    ):
        self.layers = NetList.as_net_list(layers)
        self.layer_connections: np.ndarray = LayerConnections.to_np(layer_connections, len(self.layers))

        self.num_layers = len(self.layers)

        for i, layer in enumerate(self.layers):
            for dim in definite_dimensions:
                if not layer.in_shape.is_definite(dim):
                    raise TensorShapeError(f'Dimension {dim} of in shape of layer {i} ({layer}) is indefinite but '
                                           f'required to be definite')
                if not layer.out_shape.is_definite(dim):
                    raise TensorShapeError(f'Dimension {dim} of out shape of layer {i} ({layer}) is indefinite but '
                                           f'required to be definite')

        in_shape, out_shape = LayeredNet.find_in_out_shapes(
            self.layers,
            self.layer_connections,
            combination_method,
        )
        Net.__init__(
            self,
            in_shape=in_shape,
            out_shape=out_shape,
        )

    @staticmethod
    def find_in_out_shapes(
            layers: NetList,
            layer_connections: np.ndarray,
            combination_method: ShapeCombinationMethod,
    ) -> tuple[TensorShape, TensorShape]:
        in_shapes = [layers[0].in_shape]

        for tensor_layer in range(0, len(layers) + 1):
            incoming_tensor_layers = LayeredNet.find_incoming_tensor_layers(tensor_layer, layer_connections)
            incoming_tensor_shapes: list[TensorShape] = [
                in_shapes[incoming_tensor_layer]
                for incoming_tensor_layer
                in incoming_tensor_layers
            ]

            try:
                combined_tensor_shape = LayeredNet.combine_shapes(incoming_tensor_shapes, combination_method)

                if tensor_layer < len(layers):
                    layer_out_shape = layers[tensor_layer].forward_shape(combined_tensor_shape)
                    in_shapes.append(layer_out_shape)
                else:
                    return in_shapes[0], combined_tensor_shape
            except TensorShapeError as tse:
                raise TensorShapeError(f'Error while finding shapes for tensor layer {tensor_layer}, '
                                       f'incoming tensor layers = {list(incoming_tensor_layers)}: \n' + tse.message,
                                       **tse.shapes, parent_error=tse)

    @staticmethod
    def find_incoming_tensor_layers(tensor_layer: int, layer_connections: np.ndarray) -> Iterable[int]:
        incoming_tensor_layers = layer_connections[layer_connections[:, 1] == tensor_layer][:, 0].tolist()
        return sorted(incoming_tensor_layers, reverse=True)

    @staticmethod
    def combine_shapes(
            shapes: list[TensorShape],
            combination_method: ShapeCombinationMethod
    ) -> TensorShape:
        combined_shape = shapes[0]

        if combination_method is None and len(shapes) > 1:
            raise TensorShapeError(f'Combination method is set to None, but got multiple tensor shapes to combine',
                                   **dict(enumerate(shapes)))

        for shape in shapes[1:]:
            for dim in shape.dimensions:
                if dim == 'features':
                    if combination_method == 'additive' and shape['features'] != combined_shape['features']:
                        raise TensorShapeError('Shapes can not be combined via additive method because they '
                                               'have a different size in the features dimension',
                                               **dict(enumerate(shapes)))
                    if combination_method == 'dense':
                        combined_shape['features'] += shape['features']
                else:
                    if combined_shape[dim] != shape[dim]:
                        raise TensorShapeError(f'Shapes do not all have the same size in dimension {dim}',
                                               **dict(enumerate(shapes)))

        return combined_shape

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
