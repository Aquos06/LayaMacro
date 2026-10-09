"""The Laya graft as trained and released: construction and its five nutrition questions.

Shared by training (scripts/train/train_laya.py) and inference (layamacro.inference), so a
saved model is always rebuilt exactly the way it was trained.
"""

from __future__ import annotations

from typing import Any

from layamacro.model import VisionLaya
from layamacro.schema import TARGETS, MacroBins, Question

#: Laya's checkpoint the graft starts from (Apache-2.0, convaiinnovations).
LAYA_CHECKPOINT = "convaiinnovations/laya-typed-decisions"

#: One `score` question per target; the options are the target's bin ranges.
INSTRUCTIONS = {
    "kcal": "How many kilocalories does all the food in this photo contain?",
    "protein_g": "How many grams of protein does all the food in this photo contain?",
    "carbs_g": "How many grams of carbohydrate does all the food in this photo contain?",
    "fat_g": "How many grams of fat does all the food in this photo contain?",
    "mass_g": "How much does all the food in this photo weigh, in grams?",
}

#: Laya's question-type ids (`choice`, `score`), as its decision head expects them.
QTYPE_CHOICE, QTYPE_SCORE = 0, 1


def level_labels(b: MacroBins) -> list[str]:
    """Readable option text for a target's bins; the bin edges themselves are unchanged."""
    out = []
    for i in range(b.n_bins):
        lo, hi = b.edges[i], b.edges[i + 1]
        out.append(
            f"over {lo:.3g} {b.unit}" if i == b.n_bins - 1 else f"{lo:.3g} to {hi:.3g} {b.unit}"
        )
    return out


def nutrition_questions(bins: dict[str, MacroBins]) -> list[Question]:
    return [
        Question(id=t, kind="score", instructions=INSTRUCTIONS[t], options=level_labels(bins[t]))
        for t in TARGETS
    ]


def build_graft(
    agent: Any, d_feat: int, n_patches: int, no_resampler: bool = True, n_image_tokens: int = 64
) -> VisionLaya:
    """The graft around a loaded `laya.Agent`.

    `no_resampler` (the released configuration): every SigLIP 2 patch token reaches Laya and
    only the projector is new. Otherwise a Perceiver resampler compresses them first.
    """
    if no_resampler:
        from layamacro.sequence import default_placeholder_id
        from layamacro.vision.projector import Projector
        from layamacro.vision.resampler import IdentityResampler

        lm = agent.model
        d = lm.encoder.config.hidden_size
        return VisionLaya(
            laya=lm,
            resampler=IdentityResampler(d_feat, n_patches),  # type: ignore[arg-type]
            projector=Projector(
                d_feat,
                d,
                target_std=float(lm.encoder.get_input_embeddings().weight.detach().std()),
            ),
            n_image_tokens=n_patches,
            placeholder_id=default_placeholder_id(agent.tok, lm.encoder.config.vocab_size),
        )
    return VisionLaya.from_agent(agent, encoder_dim=d_feat, n_image_tokens=n_image_tokens)
