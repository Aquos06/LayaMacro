"""Frozen vision encoder → patch embeddings.

Nothing in this module trains; to fine-tune SigLIP 2's last blocks, `vision/tail.py`
re-runs them on cached hidden states. SigLIP 2 is the default backbone
(docs/architecture.md).

**The trap this module exists to prevent:** different towers prepend a different number of
non-patch tokens, and slicing the wrong count silently corrupts every patch afterwards.

| family | prefix tokens | note |
|---|---|---|
| SigLIP / SigLIP 2 | **0** | attention-pooling (MAP) head; `last_hidden_state` is pure patches |
| DINOv2 | 1 (+ registers) | `[CLS, reg*N, patches]`; registers carry no positional embedding |
| DINOv3 | 1 + 4 registers | RoPE, no learned positional embedding |

So the count is derived from the config rather than defaulted. Guessing here produces a
subtly wrong feature map that trains fine and is wrong forever.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

#: Dimension-matched to Laya's 1024-wide ModernBERT-large, 576 patch tokens, no CLS and no
#: separate visual projection — so the projector has no width change to learn.
SIGLIP2_LARGE_384 = "google/siglip2-large-patch16-384"

#: An alternative backbone, supported but not used by the released models.
DINOV2_BASE = "facebook/dinov2-base"

#: model_type → number of leading non-patch tokens to drop. `None` means "read
#: num_register_tokens from the config and add 1 for CLS".
_PREFIX_TOKENS: dict[str, int | None] = {
    "siglip": 0,
    "siglip2": 0,
    "siglip_vision_model": 0,
    "siglip2_vision_model": 0,
    "dinov2": None,
    "dinov3_vit": None,
    "dinov3": None,
}


class FrozenVisionEncoder(nn.Module):
    """Wraps a pretrained vision tower and returns patch tokens only.

    Args:
        model_id: HuggingFace id of the vision model.
        n_prefix_tokens: leading non-patch tokens to drop. Leave `None` to derive it from
            the model config; pass an explicit int for an unrecognized tower.
        image_size: override the processor's resolution. Needed for DINOv2, whose
            `preprocessor_config.json` says 224 while `config.image_size` says 518 — the
            default pipeline therefore yields 256 patch tokens instead of 1369.
    """

    def __init__(
        self,
        model_id: str = SIGLIP2_LARGE_384,
        n_prefix_tokens: int | None = None,
        image_size: int | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_id)
        # SigLIP-style checkpoints wrap the tower; unwrap to the vision side.
        self.model = getattr(model, "vision_model", model)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model_id = model_id
        self.image_size = image_size or int(getattr(self.model.config, "image_size", 0)) or None
        self.n_prefix_tokens = (
            n_prefix_tokens if n_prefix_tokens is not None else self._infer_prefix_tokens()
        )

    def _infer_prefix_tokens(self) -> int:
        """Derive the non-patch token count from the config, or refuse to guess."""
        cfg = self.model.config
        model_type = str(getattr(cfg, "model_type", "")).lower()

        for key, value in _PREFIX_TOKENS.items():
            if model_type.startswith(key):
                if value is not None:
                    return value
                # CLS plus however many register tokens this variant carries.
                return 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)

        raise ValueError(
            f"unknown vision tower model_type={model_type!r} for {self.model_id}: cannot infer "
            "how many leading non-patch tokens to drop. Pass n_prefix_tokens explicitly — "
            "guessing silently corrupts the patch grid."
        )

    @property
    def dim(self) -> int:
        return int(self.model.config.hidden_size)

    def train(self, mode: bool = True) -> FrozenVisionEncoder:
        """Stay in eval mode permanently.

        A parent `.train()` would otherwise re-enable dropout inside a module whose weights
        are frozen — a silent train/eval discrepancy.
        """
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, pixel_values: Tensor) -> Tensor:
        """[B, 3, H, W] → [B, P, dim] patch tokens."""
        out = self.model(pixel_values=pixel_values).last_hidden_state
        return out[:, self.n_prefix_tokens :] if self.n_prefix_tokens else out
