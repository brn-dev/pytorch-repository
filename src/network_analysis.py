from typing import Literal

from torch import nn


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gradients_per_layer(model: nn.Module, param_type: Literal['all', 'weight', 'bias'] = 'all') -> None:
    for param in model.named_parameters():
        if param_type == 'all' or param[0].endswith(param_type):
            yield param[0], param[1].grad
