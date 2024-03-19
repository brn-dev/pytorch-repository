import torch

from src.networks.core.net import Net
from src.networks.core.tensor_shape import TensorShape


def find_permutation(from_order: list[str], to_order: list[str]) -> list[int]:
    if not set(from_order) == set(to_order):
        raise ValueError(f"from_order ({from_order}) does not contain the same elements as to_order ({to_order}")

    return [from_order.index(p) for p in to_order]


class PermuteDimensions(Net):

    def __init__(self, from_order: list[str], to_order: list[str]):
        in_shape = TensorShape(**{dim: None for dim in from_order})
        out_shape = in_shape.copy()
        super().__init__(
            in_shape=in_shape,
            out_shape=out_shape,
            allow_extra_dimensions=False,
        )

        self.nr_dimensions = len(from_order)
        self.permutation = find_permutation(from_order, to_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.permute(x, self.permutation)
        return x





