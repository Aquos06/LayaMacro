"""The last k blocks of SigLIP2, trainable, run on cached hidden states.

The frozen first 24 - k blocks are run once by `scripts/data/cache_features.py
--hidden-layer 24-k`; this module finishes the forward pass (blocks 24-k .. 23, then `post_layernorm`) so
its output has the same meaning as the frozen encoder's patch tokens, and can be trained.

With the weights untouched, `SiglipTail(k)(cache_L{24-k})` must reproduce the plain
`siglip2.pt` cache: `check_against` asserts that before any training run trusts it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from layamacro.vision.encoder import SIGLIP2_LARGE_384


class SiglipTail(nn.Module):
    def __init__(self, k: int, model_id: str = SIGLIP2_LARGE_384) -> None:
        super().__init__()
        from transformers import AutoModel

        vision = AutoModel.from_pretrained(model_id).vision_model
        n = len(vision.encoder.layers)
        if not 0 < k <= n:
            raise ValueError(f"k must be in 1..{n}, got {k}")
        self.k, self.first = k, n - k
        self.layers = nn.ModuleList(vision.encoder.layers[self.first :])
        self.norm = vision.post_layernorm
        del vision

    def forward(self, hidden: Tensor) -> Tensor:
        """[B, 576, 1024] hidden state entering block 24 - k → final patch tokens."""
        h = hidden
        for layer in self.layers:
            if self.training and torch.is_grad_enabled():
                # Recompute activations in the backward pass instead of storing them: 8 blocks
                # at batch 64 do not fit in 16 GB otherwise. Same result, more compute.
                out = checkpoint(layer, h, None, use_reentrant=False)
            else:
                out = layer(h, attention_mask=None)
            h = out[0] if isinstance(out, tuple) else out
        return self.norm(h)

    @torch.no_grad()
    def check_against(self, hidden: Tensor, final: Tensor, tol: float = 0.05) -> float:
        """Max-abs error vs the frozen encoder's cached output, relative to its scale."""
        was = self.training
        self.eval()
        dev = next(self.parameters()).device
        with torch.autocast("cuda", dtype=torch.bfloat16):
            got = self(hidden.to(dev)).float()
        self.train(was)
        ref = final.to(dev).float()
        err = float((got - ref).abs().mean() / ref.abs().mean())
        if err > tol:
            raise AssertionError(f"tail output differs from the frozen cache: rel err {err:.3f}")
        return err
