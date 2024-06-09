from typing import Type

from torch import nn

nn_activation_classes = (
    nn.ReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.LeakyReLU,
    nn.ReLU6, nn.SELU, nn.CELU, nn.GELU, nn.Softplus, nn.Softshrink,
    nn.Softsign, nn.Tanhshrink, nn.Hardshrink, nn.Hardtanh, nn.LogSigmoid,
    nn.Softmin, nn.Softmax, nn.Softmax2d, nn.LogSoftmax, nn.AdaptiveLogSoftmaxWithLoss
)
nn_normalization_classes = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    nn.LayerNorm, nn.GroupNorm, nn.LocalResponseNorm,
    nn.CrossMapLRN2d
)
nn_dropout_classes = (
    nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout
)
nn_convolution_classes = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    # nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d
)
nn_pooling_classes = (
    nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
    nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
    # nn.FractionalMaxPool2d, nn.FractionalMaxPool3d,
    # nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
    # nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d
)
nn_padding_classes = (
    nn.ZeroPad1d, nn.ZeroPad2d, nn.ZeroPad3d,
    nn.ReflectionPad1d, nn.ReflectionPad2d, nn.ReflectionPad3d,
    nn.ReplicationPad1d, nn.ReplicationPad2d, nn.ReplicationPad3d,
    nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d
)

def is_nn_activation_module(module: nn.Module) -> bool:
    return isinstance(module, nn_activation_classes)


def is_nn_dropout_module(module: nn.Module) -> bool:
    return isinstance(module, nn_dropout_classes)


def is_nn_pooling_module(module: nn.Module) -> bool:
    return isinstance(module, nn_pooling_classes)


def is_nn_padding_module(module: nn.Module) -> bool:
    return isinstance(module, nn_padding_classes)


def is_nn_normalization_module(module: nn.Module) -> bool:
    return isinstance(module, nn_normalization_classes)


def is_nn_convolutional_module(module: nn.Module) -> bool:
    return isinstance(module, nn_convolution_classes)


def is_nn_linear_module(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear)


def is_nn_identity_module(module: nn.Module) -> bool:
    return isinstance(module, nn.Identity)


def is_nn_sequential_module(module: nn.Module) -> bool:
    return isinstance(module, nn.Sequential)
