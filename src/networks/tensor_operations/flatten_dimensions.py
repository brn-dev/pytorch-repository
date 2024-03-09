from typing import Iterable

import numpy as np
import torch
from overrides import override

from src.networks.core.net import Net
from src.networks.core.tensor_shape import TensorShape
from src.networks.tensor_operations.functional import find_permutation


class FlattenDimensions(Net):

    def __init__(self, dim_keys_to_flatten: Iterable[str], dim_order: list[str]):
        self.flattened_dim_keys = set(dim_keys_to_flatten)

        in_shape = TensorShape(**{dim_key: None for dim_key in self.flattened_dim_keys})

        out_shape = TensorShape()
        for dim_key in self.flattened_dim_keys:
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

        self.nr_dimensions = len(dim_order)
        self.permutation = find_permutation(dim_order, permutation)
        self.nr_out_dimensions = self.nr_dimensions - len(self.flattened_dim_keys)

    @override
    def forward_shape(self, in_shape: TensorShape) -> TensorShape:
        if not set(in_shape.dimension_names).issuperset(self.flattened_dim_keys):
            raise ValueError(f'in_shape ({in_shape}) does not contain all the '
                             f'dimensions to be flattened ({self.flattened_dim_keys})')

        out_shape = self.out_shape.evaluate_forward(in_shape)

        for flattened_dim_key in self.flattened_dim_keys:
            del out_shape[flattened_dim_key]

        return out_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.permute(x, self.permutation)
        x = torch.reshape(x, list(x.shape[:self.nr_out_dimensions - 1]) + [-1])
        return x





