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

    net: 'Net'

    def __init__(
            self,
            layers: NetListLike,
            layer_connections: LayerConnections.LayerConnectionsLike,
            combination_method: ShapeCombinationMethod,
            require_definite_dimensions: Iterable[str] = (),
            connection_modulators: list[list[Net | None]] = None,
    ):
        layers = NetList.as_net_list(layers)
        layer_connections = LayerConnections.to_np(layer_connections, len(layers))

        for dim in require_definite_dimensions:
            for i, layer in enumerate(layers):
                if not layer.in_shape.is_definite(dim):
                    raise TensorShapeError(f'Dimension {dim} of in_shape of layer {i} ({layer}) is indefinite but '
                                           f'required to be definite')
                if not layer.out_shape.is_definite(dim):
                    raise TensorShapeError(f'Dimension {dim} of out_shape of layer {i} ({layer}) is indefinite but '
                                           f'required to be definite')


        if connection_modulators is not None:
            LayeredNet.check_connection_modulators_structure(
                connection_modulators,
                len(layers),
                layer_connections,
            )

        in_shape, out_shape = LayeredNet.find_in_out_shapes(
            layers,
            layer_connections,
            combination_method,
            connection_modulators,
        )

        super().__init__(
            in_shape=in_shape,
            out_shape=out_shape,
        )
        self.layers = layers
        self.num_layers = len(self.layers)

        self.layer_connections = layer_connections
        self.incoming_layer_connections = [
            self.find_incoming_tensor_layer_nrs(i, self.layer_connections)
            for i in range(self.num_layers + 1)
        ]

        self.connection_modulators = connection_modulators

        if connection_modulators is not None:
            for to_idx in range(self.num_layers + 1):
                for from_idx in range(to_idx + 1):
                    self.register_module(
                        f'connection_modulators[{to_idx}][{from_idx}]',
                        connection_modulators[to_idx][from_idx]
                    )


    @staticmethod
    def check_connection_modulators_structure(
            connection_modulators: list[list[Net | None]],
            num_layers: int,
            layer_connections: np.ndarray,
    ):
        if not len(connection_modulators) == num_layers + 1:
            raise ValueError(f'incorrect number of layers in connection_modulators '
                             f'(need {num_layers + 1}, have {len(connection_modulators)}) - {connection_modulators = }')

        for i, layer_modulators in enumerate(connection_modulators):
            incoming_layer_nrs = set(LayeredNet.find_incoming_tensor_layer_nrs(i, layer_connections))

            if not len(layer_modulators) == i + 1:
                raise ValueError(f'connections_modulators must be a 2-dimensional "triangular" list of Nets. '
                                 f'{len(connection_modulators[i]) = } should be {i + 1 = }')

            for j, modulator in enumerate(layer_modulators):
                if j in incoming_layer_nrs:
                    if not isinstance(modulator, Net):
                        raise ValueError(f'connection_modulators[{i}][{j}] is not a Net '
                                         f'but {j} is an incoming connection of {i}')
                else:
                    if modulator is not None:
                        raise ValueError(f'connection_modulators[{i}][{j}] should be None since this connection '
                                         f'is not used')



    @staticmethod
    def find_in_out_shapes(
            layers: NetList,
            layer_connections: np.ndarray,
            combination_method: ShapeCombinationMethod,
            connection_modulators: list[list[Net | None]] | None,
    ) -> tuple[TensorShape, TensorShape]:
        layer_shapes = [layers[0].in_shape]

        for tensor_layer in range(0, len(layers) + 1):
            incoming_tensor_layer_nrs = LayeredNet.find_incoming_tensor_layer_nrs(tensor_layer, layer_connections)
            incoming_tensor_shapes: list[TensorShape] = [
                layer_shapes[incoming_tensor_layer]
                for incoming_tensor_layer
                in incoming_tensor_layer_nrs
            ]

            layer_in_shapes = incoming_tensor_shapes
            if connection_modulators is not None:
                layer_in_shapes = []
                for incoming_layer_nr, incoming_layer_shape in zip(incoming_tensor_layer_nrs, incoming_tensor_shapes):
                    layer_in_shapes.append(
                        connection_modulators[tensor_layer][incoming_layer_nr]
                        .forward_shape(incoming_layer_shape)
                    )

            try:
                combined_shape = LayeredNet.combine_shapes(layer_in_shapes, combination_method)

                if tensor_layer < len(layers):
                    layer_out_shape = layers[tensor_layer].forward_shape(combined_shape)
                    layer_shapes.append(layer_out_shape)
                else:
                    return layer_shapes[0], combined_shape
            except TensorShapeError as tse:
                raise TensorShapeError(f'Error while finding shapes for tensor layer {tensor_layer}, '
                                       f'incoming tensor layers = {list(incoming_tensor_layer_nrs)}: \n' + tse.message,
                                       **tse.shapes, parent_error=tse)

    @staticmethod
    def find_incoming_tensor_layer_nrs(
            tensor_layer: int,
            layer_connections: np.ndarray,
    ) -> list[int]:
        incoming_tensor_layers = layer_connections[layer_connections[:, 1] == tensor_layer][:, 0].tolist()
        return sorted(incoming_tensor_layers)

    @staticmethod
    def combine_shapes(
            shapes: list[TensorShape],
            combination_method: ShapeCombinationMethod
    ) -> TensorShape:
        combined_shape = shapes[0].copy()

        if combination_method is None and len(shapes) > 1:
            raise TensorShapeError(f'Combination method is set to None, but got multiple tensor shapes to combine',
                                   **{str(i): shape for i, shape in enumerate(shapes)})

        for shape in shapes[1:]:
            for dim in shape.dimensions:
                if dim == 'features':
                    if combination_method == 'additive' and shape['features'] != combined_shape['features']:
                        raise TensorShapeError('Shapes can not be combined via additive method because they '
                                               'have a different size in the features dimension',
                                               **{str(i): shape for i, shape in enumerate(shapes)})
                    if combination_method == 'dense':
                        combined_shape['features'] += shape['features']
                else:
                    if combined_shape[dim] != shape[dim]:
                        raise TensorShapeError(f'Shapes do not all have the same size in dimension {dim}',
                                               **{str(i): shape for i, shape in enumerate(shapes)})

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

    @classmethod
    @abc.abstractmethod
    def from_layer_provider(cls, layer_provider: LayerProvider, *args, **kwargs) -> 'LayeredNetDerived':
        raise NotImplemented


LayeredNetDerived = TypeVar('LayeredNetDerived', bound=LayeredNet)
