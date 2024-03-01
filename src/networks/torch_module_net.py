import torch
from torch import nn

from src.networks.net import Net
from src.torch_nn_modules import is_nn_activation_module, is_nn_dropout_module, is_nn_pooling_module, \
    is_nn_padding_module, is_instance_of_group, is_nn_convolutional_module, \
    is_nn_linear_module, is_nn_identity_module

class TorchModuleNet(Net):

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

            for layer in module:
                layer = Net.as_net(layer)
                if in_features == Net.IN_FEATURES_ANY and layer.in_features != Net.IN_FEATURES_ANY:
                    in_features = layer.in_features
                if layer.out_features != Net.OUT_FEATURES_SAME:
                    out_features = layer.out_features

        else:
            raise ValueError(f'Unknown module type {type(module)}')

        super().__init__(in_features, out_features)
        self.torch_module = module

    def forward(self, *args, **kwargs):
        self.torch_module(*args, **kwargs)
