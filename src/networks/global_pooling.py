import abc
from typing import Iterable

import torch
from overrides import override

from src.networks.core.net import Net
from src.networks.core.tensor_shape import TensorShape


class GlobalPooling(Net, abc.ABC):

    """
    @:param pooling_dim_indices: indices of the dimensions to be pooled. Assumes the order (Batch, Features, S0, S1, S3)
    """
    def __init__(
            self,
            dims: Iterable[int]
    ):
        pooling_dim_indices = tuple(dims)
        pooling_dim_keys: list[str] = []

        in_shape = TensorShape()
        for dim_idx in pooling_dim_indices:
            # subtracting 2 for the dimensions batch and features
            dim_key, _ = in_shape.create_structural_dimension(dim_idx - 2)
            pooling_dim_keys.append(dim_key)

        out_shape = TensorShape()
        for structural_dim in range(2, max(pooling_dim_indices) + 1):
            if structural_dim not in pooling_dim_indices:
                out_shape.create_structural_dimension(structural_dim - 2)
            else:
                dim_key, _ = out_shape.create_structural_dimension(structural_dim - 2)
                out_shape[dim_key] = TensorShape.REMOVED_DIM_VALUE


        super().__init__(in_shape, out_shape)

        self.pooling_dim_indices = dims
        self.pooling_dim_keys = pooling_dim_keys

    @abc.abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class GlobalAveragePooling(GlobalPooling):

    def forward(self, x: torch.Tensor):
        return x.mean(dim=self.pooling_dim_indices)


class GlobalMaxPooling(GlobalPooling):

    def forward(self, x: torch.Tensor):
        return x.amax(dim=self.pooling_dim_indices)
