import os

from torch import optim
from torch import nn
import torch

def get_lr(optimizer: optim.Optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def load_if_exists(path: str):
    if not os.path.exists(path):
        return None
    return torch.load(path)

def load_state_dict_if_exists(
        obj: nn.Module | optim.Optimizer,
        path: str
) -> bool:
    if not os.path.exists(path):
        return False

    obj.load_state_dict(torch.load(path))
