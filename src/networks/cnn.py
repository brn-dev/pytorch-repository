import abc
from dataclasses import dataclass
from typing import Union, Callable, Literal

import torch
from torch import nn

from .nn_base import NNBase, HyperParameters

IntOr2iTuple = Union[int, tuple[int, int]]
NormalizationLocation = Literal['pre-layer', 'pre-activation', 'post-activation', 'post-dropout', None]


@dataclass
class ConvHyperParameters(HyperParameters):
    out_channels: int
    kernel_size: IntOr2iTuple
    stride: IntOr2iTuple
    dilation: IntOr2iTuple
    groups: IntOr2iTuple
    bias: bool
    padding: Union[str, IntOr2iTuple]
    padding_mode: str = 'zeros'


@dataclass
class CNNHyperParameters(HyperParameters):
    in_channels: int
    layers_hyper_parameters: list[ConvHyperParameters]
    activation_provider: Callable[[], nn.Module] = lambda: nn.LeakyReLU()
    activate_last_layer: bool = False
    normalization_location: Literal[
        'pre-dropout', 'post-dropout', 'pre-activation', 'post-activation', None] = 'pre-activation'
    dropout_p: float = None
    dropout_last_layer: bool = False


# TODO: pooling
class CNN(NNBase, abc.ABC):
    layers: nn.Sequential

    def __init__(
            self,
            in_channels: int,
            layers_hyper_parameters: list[ConvHyperParameters],
            activation_provider: Callable[[], nn.Module] = lambda: nn.LeakyReLU(),
            activate_last_layer: bool = False,
            normalization_location: NormalizationLocation = 'pre-activation',
            dropout_p: float = None,
            dropout_last_layer: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []

        in_channels = in_channels

        for i, layer_hyper_parameters in enumerate(layers_hyper_parameters):
            is_last_layer = i == len(layers_hyper_parameters) - 1
            out_channels = layer_hyper_parameters.out_channels

            if normalization_location == 'pre-layer':
                layers.append(self._create_batch_norm(in_channels))

            layers.append(self._create_conv(in_channels, layer_hyper_parameters))

            if normalization_location == 'pre-activation':
                layers.append(self._create_batch_norm(out_channels))

            if not is_last_layer or activate_last_layer:
                layers.append(activation_provider())

            if normalization_location == 'post-activation':
                layers.append(self._create_batch_norm(out_channels))

            if NNBase.is_dropout_active(dropout_p) and (not is_last_layer or dropout_last_layer):
                layers.append(nn.Dropout(dropout_p))

            if normalization_location == 'post-dropout':
                layers.append(self._create_batch_norm(out_channels))

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # n_sequences, sequence_length, in_features = x.shape

        return self.layers(x)

    @staticmethod
    @abc.abstractmethod
    def _create_conv(in_channels: int, conv_hyper_parameters: ConvHyperParameters) -> nn.Module:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _create_batch_norm(num_features: int) -> nn.Module:
        raise NotImplementedError
