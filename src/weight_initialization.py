from typing import TypeVar

from torch import nn

LinearOrConv = TypeVar('LinearOrConv', nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)


def lecun_initialization(module: LinearOrConv, zero_bias=True) -> LinearOrConv:
    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
    if zero_bias:
        nn.init.zeros_(module.bias)
    return module


def orthogonal_initialization(module: LinearOrConv, gain: float = 1) -> LinearOrConv:
    nn.init.orthogonal_(module.weight, gain=gain)
    if module.bias is not None:
        module.bias.data.fill_(0.0)
    return module
