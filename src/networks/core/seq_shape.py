from typing import Iterable

from torch import nn

from src.networks.core.net import Net
from src.networks.core.tensor_shape import TensorShape, TensorShapeError


def find_seq_in_out_shapes(layers: Iterable[nn.Module]):
    in_shape, current_shape = TensorShape(), TensorShape()

    for i, nn_layer in enumerate(layers):
        layer: Net = Net.as_net(nn_layer)

        accepts_shape, err = layer.accepts_in_shape(current_shape)
        if not accepts_shape:
            raise TensorShapeError(
                f'Sublayer {i} ({layer}) does not accept shape {current_shape}: \n' + err.message,
                **err.shapes, parent_error=err
            )

        for dim in layer.in_shape.dimension_names:
            if layer.in_shape.is_definite(dim) and not in_shape.is_definite(dim):
                in_shape[dim] = current_shape.evaluate_backward(dim, layer.in_shape)
            elif dim not in in_shape:
                in_shape[dim] = layer.in_shape[dim]

        current_shape = layer.forward_shape(current_shape)
        if len(list(layers)) == 9:
            print(current_shape.evaluate_forward(TensorShape(features=3, s_0=224, s_1=224)))

    return in_shape, current_shape
