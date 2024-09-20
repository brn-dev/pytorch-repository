from typing import SupportsFloat, TypeVar

import torch

T = TypeVar('T')
def identity(x: T) -> T:
    return x

def antisymmetric_power(x: torch.Tensor, exponent: torch.Tensor | SupportsFloat):
    sign = torch.sign(x)
    amplitude = torch.abs(x) ** exponent
    return sign * amplitude
