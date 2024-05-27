from typing import TypeVar, Generic

import torch
from torch import nn

from src.networks.core.torch_wrappers.conv_net import compute_conv_in_out_shapes
from src.networks.core.net import Net
from src.networks.core.seq_shape import find_seq_in_out_shapes
from src.networks.core.tensor_shape import TensorShape
from src.torch_nn_modules import is_nn_activation_module, is_nn_dropout_module, is_nn_pooling_module, \
    is_nn_padding_module, is_instance_of_group, is_nn_convolutional_module, \
    is_nn_linear_module, is_nn_identity_module


ModuleType = TypeVar('ModuleType', bound=nn.Module)


class TorchNet(Net, Generic[ModuleType]):

    def __init__(self, module: ModuleType, in_shape: TensorShape, out_shape=TensorShape):
        super().__init__(
            in_shape=in_shape,
            out_shape=out_shape
        )
        self.torch_module = module
        self.forward = self.torch_module.forward


    @staticmethod
    def wrap(module: ModuleType) -> 'TorchNet[ModuleType]':
        in_shape, out_shape = TorchNet.detect_in_out_shapes(module)
        return TorchNet(
            module=module,
            in_shape=in_shape,
            out_shape=out_shape
        )

    @staticmethod
    def detect_in_out_shapes(module: ModuleType):
        if (is_nn_activation_module(module) or is_nn_dropout_module(module)
                or is_nn_pooling_module(module) or is_nn_padding_module(module)
                or is_nn_identity_module(module)):
            in_shape, out_shape = TensorShape(), TensorShape()

        elif is_instance_of_group(module, [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                           nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]):
            in_shape, out_shape = TensorShape(features=module.num_features), TensorShape(features=module.num_features)

        elif isinstance(module, nn.LayerNorm):
            num_features = module.normalized_shape

            if is_instance_of_group(num_features, [list, tuple, torch.Size]):
                if len(num_features) > 1:
                    raise NotImplementedError('layernorm with multi-dimensional shape not implemented')
                num_features = num_features[-1]

            in_shape, out_shape = TensorShape(features=num_features), TensorShape(features=num_features)

        elif is_nn_convolutional_module(module):
            # noinspection PyTypeChecker
            in_shape, out_shape = compute_conv_in_out_shapes(module)

        elif is_nn_linear_module(module):
            in_shape, out_shape = TensorShape(features=module.in_features), TensorShape(features=module.out_features)

        elif isinstance(module, nn.Sequential):
            in_shape, out_shape = find_seq_in_out_shapes(module)

        elif isinstance(module, nn.MultiheadAttention):
            shape = TensorShape(features=module.embed_dim)
            shape.create_structural_dimension()
            in_shape, out_shape = shape, shape.copy()

        else:
            raise ValueError(f'Unknown module type {type(module)}')

        return in_shape, out_shape

