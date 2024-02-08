from torch import nn

from .cnn import CNN, ConvHyperParameters


class CNN2d(CNN):

    def _create_conv(self, in_channels: int, hyper_parameters: ConvHyperParameters) -> nn.Conv2d:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=hyper_parameters.out_channels,
            kernel_size=hyper_parameters.kernel_size,
            stride=hyper_parameters.stride,
            padding=hyper_parameters.padding,
            groups=hyper_parameters.groups,
            bias=hyper_parameters.bias,
            padding_mode=hyper_parameters.padding_mode
        )

    def _create_batch_norm(self, num_features: int) -> nn.BatchNorm2d:
        return nn.BatchNorm2d(num_features)
