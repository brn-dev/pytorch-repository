from typing import Optional

import torch
from torch import nn

from src.networks.nn_base import NNBase


class MultiheadSelfAttention(NNBase):

    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim,
                                         batch_first, device, dtype)

    def forward(
            self,
            x: torch.Tensor,
            extract_attn_output: bool = True,
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False
    ) -> (torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]):
        attention_out, attention_out_weights = self.mha.forward(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if extract_attn_output:
            return attention_out
        return attention_out, attention_out_weights
