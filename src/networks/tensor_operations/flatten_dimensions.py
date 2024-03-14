from typing import Iterable

import torch
from overrides import override

from src.networks.core.net import Net
from src.networks.core.tensor_shape import TensorShape, TensorShapeError
from src.networks.tensor_operations.permute_dimensions import find_permutation


class FlattenDimensions(Net):

    def __init__(self, dim_keys_to_flatten: Iterable[str], dim_order: list[str]):
        self.dim_keys_to_flatten = set(dim_keys_to_flatten)

        in_shape = TensorShape(**{dim_key: None for dim_key in dim_order})

        out_shape = TensorShape()
        for dim_key in self.dim_keys_to_flatten:
            out_shape['features'] *= in_shape[dim_key]

        super().__init__(
            in_shape=in_shape,
            out_shape=out_shape,
        )

        permutation = list(dim_order)

        previous_index = -1
        for dim_key in dim_keys_to_flatten:
            if dim_key == 'features':
                raise ValueError('features dimension can not be flattened')

            index = permutation.index(dim_key)
            if index < previous_index:
                raise ValueError('Dim keys to flatten must appear in the same order as in the dim order')

            permutation.pop(index)
            permutation.insert(-1, dim_key)

            previous_index = index

        self.nr_in_dimensions = len(dim_order)
        self.permutation = find_permutation(dim_order, permutation)
        self.nr_out_dimensions = self.nr_in_dimensions - len(self.dim_keys_to_flatten)

    @override
    def forward_shape(self, in_shape: TensorShape) -> TensorShape:
        if not set(in_shape.dimension_names).issuperset(self.dim_keys_to_flatten):
            raise TensorShapeError(f'in_shape ({in_shape}) does not contain all the dimensions to be flattened '
                                   f'({self.dim_keys_to_flatten})', self_in_shape=self.in_shape, in_shape=in_shape)

        out_shape = self.out_shape.evaluate_forward(in_shape)

        for flattened_dim_key in self.dim_keys_to_flatten:
            del out_shape[flattened_dim_key]

        return out_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.permute(x, self.permutation)
        x = torch.flatten(x, start_dim=self.nr_out_dimensions - 1, end_dim=-1)
        return x
