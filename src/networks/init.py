from torch import nn


def lecun_initialization(linear: nn.Linear, zero_bias=True):
    nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='linear')
    if zero_bias:
        nn.init.zeros_(linear.bias)
