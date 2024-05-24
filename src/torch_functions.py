from typing import SupportsFloat

import torch


def antisymmetric_power(x: torch.Tensor, exponent: torch.Tensor | SupportsFloat):
    sign = torch.sign(x)
    amplitude = torch.abs(x) ** exponent
    return sign * amplitude
