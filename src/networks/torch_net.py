import torch
from torch import nn

from src.networks.net import Net
from src.networks.tensor_shape import TensorShape
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
            # TODO: spacial dimensions
            in_shape, out_shape = TensorShape(features=module.in_channels), TensorShape(features=module.out_channels)

        elif is_nn_linear_module(module):
            in_shape, out_shape = TensorShape(features=module.in_features), TensorShape(features=module.out_features)

        elif isinstance(module, nn.Sequential):
            # TODO: spacial dimensions
            in_shape, out_shape = TensorShape(), TensorShape()

            for nn_layer in module:
                layer: Net = TorchNet(nn_layer)
                layer_in_features_definite = layer.in_shape.is_definite('features')

                if not layer_in_features_definite:
                    in_shape = layer.out_shape.evaluate_forward(in_shape)
                elif not in_shape.is_definite('features'):
                    in_shape = in_shape.evaluate_backward(layer.in_shape)

                if layer.out_shape.is_definite('features'):
                    out_shape = layer.out_shape
                else:
                    out_shape = layer.out_shape.evaluate_forward(out_shape)

        else:
            raise ValueError(f'Unknown module type {type(module)}')

        return in_shape, out_shape
