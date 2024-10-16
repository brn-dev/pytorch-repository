import math

import torch

from src.networks.core.net import Net
from src.networks.core.tensor_shape import TensorShape


class SinusoidalPositionalEncoding(Net):

    @property
    def in_features(self) -> int:
        return 1

    @property
    def out_features(self) -> int:
        return self._out_features

    pos_embedding: torch.Tensor

    def __init__(self, d_model: int, max_len: int):
        in_shape = TensorShape(features=0)
        in_shape.create_structural_dimension()

        out_shape = TensorShape(features=d_model)
        out_shape.create_structural_dimension()

        super().__init__(
            in_shape=in_shape,
            out_shape=out_shape,
        )

        pos_embedding = torch.zeros(max_len, d_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0) / d_model)

        pos_embedding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_embedding[:, 1::2] = torch.cos(positions_list * division_term)

        self.register_buffer('pos_embedding', pos_embedding)
        self._out_features = d_model

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        embedding = self.pos_embedding[positions]
        return embedding
