"""Projector: resampled visual tokens → Laya's input-embedding space.

Deliberately ends on a plain Linear with **no LayerNorm**: ModernBERT's embedding
layer applies `drop(norm(inputs_embeds))` to whatever is injected, so normalizing
here would double-normalize. See docs/laya.md.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Projector(nn.Module):
    """[B, N, in_dim] → [B, N, out_dim], where out_dim is Laya's hidden size (1024).

    Args:
        in_dim: resampler output width (the frozen encoder's hidden size).
        out_dim: Laya's `encoder.config.hidden_size`.
        hidden: width of the single hidden layer; defaults to out_dim.
        target_std: initial output scale. Defaults to the std of Laya's own token
            embeddings, passed in by the caller, so the encoder sees visual tokens on
            roughly the same scale as text ones from step zero.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int | None = None,
        dropout: float = 0.0,
        target_std: float | None = None,
    ) -> None:
        super().__init__()
        hidden = hidden or out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        if target_std is not None:
            self._rescale_output(target_std)

    def _rescale_output(self, target_std: float) -> None:
        """Match the initial output scale to the text embeddings'.

        Without this the projector's outputs can start orders of magnitude away from
        the embedding distribution. The embedding LayerNorm absorbs a constant factor,
        but a bad starting scale still wastes early training, and can stall it entirely
        while Laya is frozen.
        """
        final = self.net[-1]
        assert isinstance(final, nn.Linear)
        with torch.no_grad():
            # Output std of a Linear with input std 1 is ~ w_std * sqrt(fan_in).
            fan_in = final.weight.shape[1]
            final.weight.mul_(target_std / (final.weight.std() * fan_in**0.5))
            final.bias.zero_()

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.shape[-1] != self.in_dim:
            raise ValueError(f"expected in_dim={self.in_dim}, got {tokens.shape[-1]}")
        return self.net(tokens)
