"""VisionLaya: the graft.

Laya's `DecisionModel` is a bare `nn.Module` whose `forward` hardcodes
`self.encoder(input_ids=...)`. The ModernBERT backbone *does* accept `inputs_embeds`;
`DecisionModel` simply never passes it (docs/laya.md). That gap is the whole surgery.

This **wraps** rather than subclasses `DecisionModel`, which buys two things:

1. `laya.load()` stays the loader for Laya's weights, so its `strict=True` prefix/shape
   audit is satisfied and we never reimplement checkpoint handling.
2. The decision head, the `[MASK]` gather and the scorer are used **as they are** —
   Laya's decision head is never modified, which is what preserves "new questions at
   request time with no retraining".

The one duplicated thing is the head math in `_decide_from_embeds`, because upstream's
`forward` cannot be reached without going through `input_ids`.
`tests/test_graft.py::test_embeds_path_matches_upstream_forward` asserts our path is
**bit-identical** to `DecisionModel.forward` on a no-image batch, which is what makes the
duplication safe: if upstream changes its head, that test fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from layamacro.schema import MACROS
from layamacro.vision.projector import Projector
from layamacro.vision.resampler import PerceiverResampler


@dataclass
class GraftOutput:
    """`logits` is per-option, exactly as upstream returns it."""

    logits: Tensor  # [B, k_max]
    act_logits: Tensor  # [B, n_act] - upstream's; carries no signal, see docs/laya.md
    pooled: Tensor  # [B, d] the [CLS] hidden state, for the regression head


class MacroRegressor(nn.Module):
    """Scalar macro head — the deliberate exception to typed decisions.

    Exists to measure what the bin framing costs in precision. Predicts **log1p** of each
    macro, because the errors that matter are relative; `predict` inverts that.
    """

    def __init__(self, dim: int, n_macros: int = len(MACROS), hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_macros),
        )

    def forward(self, pooled: Tensor) -> Tensor:
        return self.net(pooled)

    @torch.no_grad()
    def predict(self, pooled: Tensor) -> Tensor:
        """Invert the log1p target back to grams/kcal.

        Clamped at 0 because the raw output is unconstrained: `log1p(y) >= 0` for any
        valid macro, so a negative prediction means "none of it", not a negative mass.
        Only the inverse is clamped — the loss stays in log space on the raw output, so
        gradients are unaffected.
        """
        return torch.expm1(self(pooled).clamp(min=0.0, max=20.0))


class VisionLaya(nn.Module):
    """Frozen vision encoder output → resampler → projector → Laya.

    Only `resampler`, `projector` (and any heads) are new parameters. Laya's own weights
    live in `self.laya` and are loaded by `laya.load()`.
    """

    def __init__(
        self,
        laya: nn.Module,
        resampler: PerceiverResampler,
        projector: Projector,
        n_image_tokens: int,
        placeholder_id: int,
    ) -> None:
        super().__init__()
        # Annotated Any deliberately: upstream's DecisionModel is an unannotated bare
        # nn.Module, so `nn.Module.__getattr__` types every attribute as `Tensor | Module`
        # and each real access (`.encoder`, `.scorer`, `.head`) becomes a false positive.
        # It is still registered as a submodule at runtime, so parameters/state_dict work.
        self.laya: Any = laya
        self.resampler = resampler
        self.projector = projector
        self.n_image_tokens = n_image_tokens
        self.placeholder_id = placeholder_id
        #: Set True while debugging to re-check the graft invariants per batch.
        self.check_invariants = False

        if resampler.n_queries != n_image_tokens:
            raise ValueError(
                f"resampler emits {resampler.n_queries} tokens but {n_image_tokens} slots "
                "are reserved; the sequence builder and the resampler must agree"
            )
        if projector.out_dim != self.hidden_size:
            raise ValueError(
                f"projector outputs {projector.out_dim}, Laya's hidden size is {self.hidden_size}"
            )

    @property
    def hidden_size(self) -> int:
        """Laya's embedding width — what the projector must output."""
        return int(self.laya.encoder.config.hidden_size)

    @property
    def vocab_size(self) -> int:
        """Embedding-table height. Larger than the tokenizer's id space, which is what
        makes an unused placeholder id available (see sequence.default_placeholder_id)."""
        return int(self.laya.encoder.config.vocab_size)

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_agent(
        cls,
        agent,
        encoder_dim: int,
        n_image_tokens: int = 64,
        n_heads: int = 8,
        n_layers: int = 2,
    ) -> VisionLaya:
        """Build around an already-loaded `laya.Agent`.

        The projector's initial output scale is matched to Laya's own token embeddings,
        so the encoder sees visual tokens on a familiar scale from step zero.
        """
        from layamacro.sequence import default_placeholder_id

        laya_model = agent.model
        d = laya_model.encoder.config.hidden_size
        emb_std = float(laya_model.encoder.get_input_embeddings().weight.detach().std())
        return cls(
            laya=laya_model,
            resampler=PerceiverResampler(
                encoder_dim, n_queries=n_image_tokens, n_heads=n_heads, n_layers=n_layers
            ),
            projector=Projector(encoder_dim, d, target_std=emb_std),
            n_image_tokens=n_image_tokens,
            placeholder_id=default_placeholder_id(agent.tok, laya_model.encoder.config.vocab_size),
        )

    # -- parameter groups -----------------------------------------------------

    def new_parameters(self) -> list[nn.Parameter]:
        """The from-scratch parameters (resampler + projector): all that scope "new" trains."""
        return list(self.resampler.parameters()) + list(self.projector.parameters())

    def freeze_laya(self, frozen: bool = True) -> None:
        """Freeze (or unfreeze) all of Laya's own weights."""
        self.laya.requires_grad_(not frozen)

    def set_trainable(self, scope: str) -> list[nn.Parameter]:
        """Select what trains, and return the parameters to hand an optimizer.

        - ``"new"``  — resampler + projector only. The strictest setting: the
          projector must steer a *frozen* readout, which is the real alignment question.
        - ``"head"`` — also Laya's decision head, scorer, type embedding and act head,
          but not the 28-layer backbone. Useful to separate "the graft cannot carry
          information" from "the frozen readout is the bottleneck" — those look
          identical in a loss curve and have completely different fixes.
        - ``"all"``  — everything, Laya's backbone included (full fine-tune).
        """
        if scope not in {"new", "head", "all"}:
            raise ValueError(f"scope must be new/head/all, got {scope!r}")

        self.laya.requires_grad_(scope == "all")
        for p in self.new_parameters():
            p.requires_grad_(True)

        if scope == "head":
            for mod in (self.laya.head, self.laya.scorer, self.laya.type_emb, self.laya.act_head):
                if mod is not None:
                    mod.requires_grad_(True)

        return [p for p in self.parameters() if p.requires_grad]

    def param_groups(
        self, scope: str, lr_new: float = 3e-4, lr_pretrained: float = 1e-4
    ) -> list[dict]:
        """Optimizer groups with separate LRs for fresh vs pretrained weights.

        Freshly initialized modules need a much larger step than pretrained ones. A
        single LR high enough for the resampler will destroy Laya's decision head within
        a few steps — the symptom is loss collapsing to exactly ``ln(n_options)`` and
        staying there, because the head degenerates into a constant output. Upstream
        makes the same split (encoder 2.5e-5, head 1e-4; see docs/laya.md).
        """
        self.set_trainable(scope)
        new_ids = {id(p) for p in self.new_parameters()}
        new = [p for p in self.parameters() if p.requires_grad and id(p) in new_ids]
        pretrained = [p for p in self.parameters() if p.requires_grad and id(p) not in new_ids]

        groups = [{"params": new, "lr": lr_new}]
        if pretrained:
            groups.append({"params": pretrained, "lr": lr_pretrained})
        return groups

    def enable_gradient_checkpointing(self) -> None:
        """Upstream's memory strategy (its recipe is a full fine-tune, not LoRA)."""
        self.laya.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.laya.head_checkpointing = True

    # -- forward --------------------------------------------------------------

    def encode_image(self, patch_embeds: Tensor) -> Tensor:
        """[B, P, d_v] frozen patch tokens → [B, n_image_tokens, d] soft tokens."""
        return self.projector(self.resampler(patch_embeds))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        marker_pos: Tensor,
        marker_mask: Tensor,
        qtype: Tensor,
        image_tokens: Tensor | None = None,
        image_slots: Tensor | None = None,
    ) -> GraftOutput:
        """`image_tokens` are already projected — pass `encode_image(...)` output.

        Encode once, reuse across every question about the same image: that
        amortization is what keeps several questions per photo cheap.
        """
        embeds = self.laya.encoder.get_input_embeddings()(input_ids)

        if image_tokens is not None:
            if image_slots is None:
                raise ValueError("image_tokens given without image_slots")
            if self.check_invariants:
                self._assert_slots(input_ids, image_slots, marker_pos)
            # Out-of-place scatter: differentiable w.r.t. image_tokens, and never
            # mutates the embedding table's output in place.
            idx = image_slots.unsqueeze(-1).expand(-1, -1, embeds.shape[-1])
            embeds = embeds.scatter(1, idx, image_tokens.to(embeds.dtype))

        return self._decide_from_embeds(embeds, attention_mask, marker_pos, marker_mask, qtype)

    def _assert_slots(self, input_ids: Tensor, image_slots: Tensor, marker_pos: Tensor) -> None:
        """Re-check the invariants that fail silently. Debug only."""
        at_slots = input_ids.gather(1, image_slots)
        if not bool((at_slots == self.placeholder_id).all()):
            raise AssertionError("an image slot does not hold the placeholder token")
        if image_slots.shape[1] != self.n_image_tokens:
            raise AssertionError(
                f"expected {self.n_image_tokens} slots, got {image_slots.shape[1]}"
            )
        # A marker inside the image run would mean an option's logit is read off a
        # visual token.
        overlap = (marker_pos.unsqueeze(-1) == image_slots.unsqueeze(1)).any()
        if bool(overlap):
            raise AssertionError("a marker_pos falls inside the image slots")

    def _decide_from_embeds(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        marker_pos: Tensor,
        marker_mask: Tensor,
        qtype: Tensor,
    ) -> GraftOutput:
        """Mirrors `DecisionModel.forward`, differing only in taking `inputs_embeds`.

        Kept line-for-line faithful on purpose; `tests/test_graft.py` asserts exact
        equality with upstream so drift is caught rather than absorbed.
        """
        m = self.laya
        h = m.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
        h = h + m.type_emb(qtype)[:, None, :]
        if m.head is not None:
            pad = ~attention_mask.bool()
            for layer in m.head.layers:
                if m.head_checkpointing and self.training and torch.is_grad_enabled():
                    h = checkpoint(layer, h, src_key_padding_mask=pad, use_reentrant=False)
                else:
                    h = layer(h, src_key_padding_mask=pad)

        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        gathered = torch.gather(h, 1, idx)
        logits = m.scorer(gathered).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = m.act_head(torch.cat([pooled, feats], -1))
        return GraftOutput(logits=logits, act_logits=act_logits, pooled=pooled)

    # -- persistence ----------------------------------------------------------
    #
    # Laya's own weights are NOT saved here: `laya.load()` validates strict=True plus a
    # prefix/shape audit and rejects a checkpoint carrying our modules (docs/laya.md).
    # So these methods save only the new modules, to reattach to a freshly loaded Laya.
    # When Laya's own weights train too (scope "head" / "all"), save the whole
    # `state_dict()` and load it into a rebuilt graft, as scripts/train/train_laya.py does.

    def save_new_modules(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "resampler": self.resampler.state_dict(),
                "projector": self.projector.state_dict(),
                "n_image_tokens": self.n_image_tokens,
                "placeholder_id": self.placeholder_id,
                "resampler_dim": self.resampler.dim,
            },
            p,
        )

    def load_new_modules(self, path: str | Path, strict: bool = True) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt["n_image_tokens"] != self.n_image_tokens:
            raise ValueError(
                f"checkpoint reserves {ckpt['n_image_tokens']} image tokens, "
                f"this model uses {self.n_image_tokens}"
            )
        if ckpt["resampler_dim"] != self.resampler.dim:
            raise ValueError(
                f"checkpoint resampler dim {ckpt['resampler_dim']} != {self.resampler.dim} "
                "— a different vision encoder produced it"
            )
        self.resampler.load_state_dict(ckpt["resampler"], strict=strict)
        self.projector.load_state_dict(ckpt["projector"], strict=strict)
        self.placeholder_id = ckpt["placeholder_id"]
