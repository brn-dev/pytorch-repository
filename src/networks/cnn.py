import abc
from dataclasses import dataclass
from typing import Union, Callable, Literal

import torch
from torch import nn

from .nn_base import NNBase
from ..hyper_parameters import HyperParameters

IntOr2iTuple = Union[int, tuple[int, int]]
NormalizationLocation = Literal['pre-dropout', 'post-dropout', 'pre-activation', 'post-activation', None]


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
    normalization_location: Literal[
        'pre-dropout', 'post-dropout', 'pre-activation', 'post-activation', None] = 'pre-activation'
    activation_provider: Callable[[], nn.Module] = lambda: nn.LeakyReLU()
    dropout: float = 0.0


# TODO: pooling
class CNN(NNBase, abc.ABC):
    layers: nn.Sequential

    def __init__(
            self,
            in_channels: int,
            layers_hyper_parameters: list[ConvHyperParameters],
            normalization_location: NormalizationLocation = 'pre-activation',
            activation_provider: Callable[[], nn.Module] = lambda: nn.LeakyReLU(),
            dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []

        in_channels = in_channels

        print(layers_hyper_parameters)
        for layer_hyper_parameters in layers_hyper_parameters:
            print(layer_hyper_parameters)
            out_channels = layer_hyper_parameters.out_channels

            if normalization_location == 'pre-dropout':
                layers.append(self._create_batch_norm(in_channels))

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            if normalization_location == 'post-dropout':
                layers.append(self._create_batch_norm(in_channels))

            layers.append(self._create_conv(in_channels, layer_hyper_parameters))

            if normalization_location == 'pre-activation':
                layers.append(self._create_batch_norm(in_channels))

            layers.append(activation_provider())

            if normalization_location == 'post-activation':
                layers.append(self._create_batch_norm(in_channels))

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # n_sequences, sequence_length, in_features = x.shape

        return self.layers.forward(x)

    @staticmethod
    @abc.abstractmethod
    def _create_conv(in_channels: int, conv_hyper_parameters: ConvHyperParameters) -> nn.Module:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _create_batch_norm(num_features: int) -> nn.Module:
        raise NotImplementedError

    @classmethod
    def from_hyper_parameters(cls, hyper_parameters: CNNHyperParameters):
        return cls(
            in_channels=hyper_parameters.in_channels,
            layers_hyper_parameters=hyper_parameters.layers_hyper_parameters,
            normalization_location=hyper_parameters.normalization_location,
            activation_provider=hyper_parameters.activation_provider,
            dropout=hyper_parameters.dropout,
        )
