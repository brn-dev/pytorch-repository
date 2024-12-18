from typing import Literal

import numpy as np
import torch
from torch import nn


def count_parameters(model: nn.Module, requires_grad_only: bool = True):
    return sum(p.numel() for p in model.parameters() if not requires_grad_only or p.requires_grad)


def get_gradients_per_parameter(model: nn.Module, param_type: Literal['all', 'weight', 'bias'] = 'all'):
    for name, param in model.named_parameters():
        if param_type == 'all' or name.endswith(param_type):
            yield name, param.grad


def calculate_grad_norm(model: nn.Module) -> float:
    return np.sqrt(sum([torch.norm(p)**2 for p in model.parameters()]))
