from typing import Iterable

import numpy as np
import torch
from overrides import override

from src.networks.core.net import Net
from src.networks.core.tensor_shape import TensorShape
from src.networks.tensor_operations.functional import find_permutation


class PermuteDimensions(Net):

    def __init__(self, from_order: list[str], to_order: list[str]):
        in_shape = TensorShape(**{dim: None for dim in from_order})
        out_shape = in_shape.copy()
        super().__init__(
            in_shape=in_shape,
            out_shape=out_shape,
        )

        self.nr_dimensions = len(from_order)
        self.permutation = find_permutation(from_order, to_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.permute(x, self.permutation)
        return x





