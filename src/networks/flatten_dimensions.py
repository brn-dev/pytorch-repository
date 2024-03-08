from typing import Iterable

import numpy as np
from overrides import override

from src.networks.core.net import Net
from src.networks.core.tensor_shape import TensorShape


class FlattenDimensions(Net):

    def __init__(self, *dim_keys: str):
        self.flattened_dim_keys = set(dim_keys)

        in_shape = TensorShape(**{dim_key: None for dim_key in dim_keys})

        out_shape = TensorShape()
        for dim_key in dim_keys:
            out_shape['features'] *= in_shape[dim_key]

        super().__init__(
            in_shape=in_shape,
            out_shape=out_shape,
        )

    @override
    def forward_shape(self, in_shape: TensorShape) -> TensorShape:
        if not set(in_shape.dimension_names).issuperset(self.flattened_dim_keys):
            raise ValueError(f'in_shape ({in_shape}) does not contain all the '
                             f'dimensions to be flattened ({self.flattened_dim_keys})')

        out_shape = self.out_shape.evaluate_forward(in_shape)

        for flattened_dim_key in self.flattened_dim_keys:
            del out_shape[flattened_dim_key]

        return out_shape


