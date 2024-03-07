import torch
from torch import nn

import sympy as sp

from src.networks.core.net import Net
from src.networks.core.seq_shape import find_seq_in_out_shapes
from src.networks.core.tensor_shape import TensorShape
from src.torch_nn_modules import is_nn_activation_module, is_nn_dropout_module, is_nn_pooling_module, \
    is_nn_padding_module, is_instance_of_group, is_nn_convolutional_module, \
    is_nn_linear_module, is_nn_identity_module


class TorchNet(Net):

    def __init__(self, module: nn.Module):
        in_shape, out_shape = self.detect_in_out_shapes(module)

        super().__init__(
            in_shape=in_shape,
            out_shape=out_shape,
        )
        self.torch_module = module


    def forward(self, *args, **kwargs):
        return self.torch_module(*args, **kwargs)


    @staticmethod
    def detect_in_out_shapes(module: nn.Module):
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
                    raise NotImplementedError('layernorm with complex shape not implemented')
                num_features = num_features[-1]

            in_shape, out_shape = TensorShape(features=num_features), TensorShape(features=num_features)

        elif is_nn_convolutional_module(module):
            # noinspection PyTypeChecker
            in_shape, out_shape = TorchNet.compute_conv_in_out_shapes(module)

        elif is_nn_linear_module(module):
            in_shape, out_shape = TensorShape(features=module.in_features), TensorShape(features=module.out_features)

        elif isinstance(module, nn.Sequential):
            in_shape, out_shape = find_seq_in_out_shapes(module)

        else:
            raise ValueError(f'Unknown module type {type(module)}')

        return in_shape, out_shape

    @staticmethod
    def compute_conv_in_out_shapes(conv: nn.Conv1d | nn.Conv2d | nn.Conv3d):
        in_shape, out_shape = TensorShape(features=conv.in_channels), TensorShape(features=conv.out_channels)


        for conv_dim_nr in range(len(conv.kernel_size)):
            kernel_size = conv.kernel_size[conv_dim_nr]
            stride = conv.stride[conv_dim_nr]
            padding = conv.padding[conv_dim_nr]
            dilation = conv.dilation[conv_dim_nr]

            dim_key, dim_symbol = out_shape.create_structural_dimension()
            out_shape[dim_key] = sp.floor(
                (dim_symbol + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
            )

        return in_shape, out_shape

