from typing import Optional

import torch
from overrides import override
from torch import nn

from src.networks.core.net import Net
from src.networks.core.tensor_shape import TensorShape, TensorShapeError


class MultiheadSelfAttention(Net):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.,
            bias: bool = True,
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            kdim: int = None,
            vdim: int = None,
            batch_first: bool = False,
            device=None,
            dtype=None,
    ):
        shape = TensorShape(features=embed_dim)
        shape.create_structural_dimension()

        super().__init__(shape, shape.copy(), allow_extra_dimensions=False)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim,
                                         batch_first, device, dtype)

    @override
    def check_in_shape(self, in_shape: TensorShape):
        super().check_in_shape(in_shape)
        if len(in_shape.dimension_names) > 3:
            raise TensorShapeError(f'MultiheadSelfAttention requires exactly 3 input '
                                   f'dimensions (S, B, F), got {in_shape}')

    def forward(
            self,
            x: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
    ) -> (torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]):
        attention_out, attention_out_weights = self.mha(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if need_weights:
            return attention_out, attention_out_weights
        return attention_out
