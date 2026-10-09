"""Perceiver-style resampler: many patch tokens → a fixed small number of tokens.

A frozen encoder emits hundreds of patch tokens; Laya's sequence budget wants 32-64
(docs/architecture.md). Learned queries cross-attend to the patches, so the output
count is fixed and independent of input resolution.

32 vs 64 is an ablation, hence `n_queries`. The released Laya models skip the resampler
(`IdentityResampler`, `--no-resampler`).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + FFN. Queries attend to the patch tokens."""

    def __init__(self, dim: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.attn(self.norm_q(q), kv_n, kv_n, need_weights=False)
        q = q + attn_out
        return q + self.ff(self.norm_ff(q))


class PerceiverResampler(nn.Module):
    """[B, P, dim] patch tokens → [B, n_queries, dim].

    Args:
        dim: the frozen encoder's hidden size.
        n_queries: output token count (the 32 vs 64 ablation).
        n_heads: attention heads; must divide `dim`.
        n_layers: cross-attention blocks.
    """

    def __init__(
        self,
        dim: int,
        n_queries: int = 64,
        n_heads: int = 8,
        n_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"n_heads={n_heads} must divide dim={dim}")
        self.dim = dim
        self.n_queries = n_queries
        # Scaled init: unit-ish variance after the first LayerNorm.
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * dim**-0.5)
        self.blocks = nn.ModuleList(
            _CrossAttentionBlock(dim, n_heads, ff_mult, dropout) for _ in range(n_layers)
        )
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, patch_embeds: Tensor) -> Tensor:
        if patch_embeds.ndim != 3:
            raise ValueError(f"expected [B, P, dim], got {tuple(patch_embeds.shape)}")
        if patch_embeds.shape[-1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {patch_embeds.shape[-1]}")
        q = self.queries.unsqueeze(0).expand(patch_embeds.shape[0], -1, -1)
        for block in self.blocks:
            q = block(q, patch_embeds)
        return self.norm_out(q)


class IdentityResampler(nn.Module):
    """No resampling: every patch token goes to Laya (via the projector).

    Started as the diagnostic for "is the resampler the bottleneck?"; the released Laya
    models use it. With SigLIP2's
    576 patches the image run is 576 slots, which fits Laya's 1,024-token context next to an
    18-option question (~250 tokens).
    """

    def __init__(self, dim: int, n_tokens: int) -> None:
        super().__init__()
        self.dim = dim
        self.n_queries = n_tokens

    def forward(self, patch_embeds: Tensor) -> Tensor:
        if patch_embeds.shape[1] != self.n_queries or patch_embeds.shape[-1] != self.dim:
            raise ValueError(
                f"expected [B, {self.n_queries}, {self.dim}], got {tuple(patch_embeds.shape)}"
            )
        return patch_embeds
