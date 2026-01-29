"""Attention utilities for slot-based MAR models."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class QKNormalizedMultiheadAttention(nn.Module):
    """
    Multi-head attention layer that normalises queries and keys to unit norm
    before computing their similarity scores.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        batch_first: bool = False,
        bias: bool = True,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.qk_norm = qk_norm
        self.norm_eps = norm_eps
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
        bsz = tensor.shape[0]
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        bsz, tgt_len, _ = query.shape
        src_len = key.shape[1]

        q = self._shape(self.q_proj(query), tgt_len)
        k = self._shape(self.k_proj(key), src_len)
        v = self._shape(self.v_proj(value), src_len)

        if self.qk_norm:
            q = F.normalize(q, dim=-1, eps=self.norm_eps)
            k = F.normalize(k, dim=-1, eps=self.norm_eps)
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        else:
            attn_scores = torch.matmul(q * self.scale, k.transpose(-1, -2))

        min_value = torch.finfo(attn_scores.dtype).min

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                expanded_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                expanded_mask = attn_mask.unsqueeze(1)
            else:
                raise ValueError("attn_mask must be 2D or 3D.")
            expanded_mask = expanded_mask.to(device=attn_scores.device)
            if expanded_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(expanded_mask, min_value)
            else:
                attn_scores = attn_scores + expanded_mask.to(attn_scores.dtype)

        if key_padding_mask is not None:
            padding = key_padding_mask.to(torch.bool).unsqueeze(1).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(padding, min_value)

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(context)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn.mean(dim=1)
            else:
                attn_weights = attn
        else:
            attn_weights = None

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_weights
