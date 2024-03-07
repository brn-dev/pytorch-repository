from typing import Iterable

from torch import nn

from src.networks.core.net import Net
from src.networks.core.tensor_shape import TensorShape


def find_seq_in_out_shapes(layers: Iterable[nn.Module]):
    in_shape, current_shape = TensorShape(), TensorShape()

    for i, nn_layer in enumerate(layers):
        layer: Net = Net.as_net(nn_layer)
        if not layer.accepts_shape(current_shape):
            raise ValueError(f'Sublayer {i} ({layer}) does not accept shape {current_shape}')

        for definite_dim in layer.in_shape.definite_dimension_names:
            if not in_shape.is_definite(definite_dim):
                in_shape[definite_dim] = current_shape.evaluate_backward(definite_dim, layer.in_shape)

        current_shape = layer.forward_shape(current_shape)

    return in_shape, current_shape
