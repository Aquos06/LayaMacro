"""Typed questions, answers, and the macro target definition.

Format conversion only — no model logic. Laya has *two* question formats and mixing
them up fails silently, so the conversion lives in exactly one place:

- public  ``{"type", "instructions", "criteria"}``  → ``agent.predict``
- internal ``{"t", "ins", "crit"}``                 → ``laya.common.build_sequence``

See docs/laya.md.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

QuestionKind = Literal["choice", "score", "noul"]

#: The four macro targets, as whole-meal totals.
MACROS: tuple[str, ...] = ("kcal", "protein_g", "carbs_g", "fat_g")

#: Everything a model outputs: the macros plus total plate mass, which all three training
#: datasets weigh directly. Portion is the dominant error term, so it is worth predicting
#: on its own. MACROS stays the 4-tuple for code that needs only the macros.
TARGETS: tuple[str, ...] = (*MACROS, "mass_g")


class Question(BaseModel):
    """One typed question, convertible to either Laya format."""

    id: str
    kind: QuestionKind
    instructions: str
    #: choice → {key: description}; score → ordered level labels; noul → None
    options: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def _check_options(self) -> Question:
        if self.kind == "choice":
            if not self.options:
                raise ValueError(f"{self.id}: choice needs a non-empty options mapping")
        elif self.kind == "score":
            if not isinstance(self.options, list) or not self.options:
                raise ValueError(f"{self.id}: score needs an ordered, non-empty list of levels")
        elif self.options is not None and set(self.options) - {"true", "false"}:
            # Upstream raises on any other key; catch it here with a better message.
            raise ValueError(f"{self.id}: noul options may only be keyed 'true'/'false'")
        return self

    @property
    def n_options(self) -> int:
        if self.kind == "noul":
            return 2
        assert self.options is not None
        return len(self.options)

    @property
    def calibration_key(self) -> tuple[str, int]:
        """Temperatures are fitted per (kind, option count); see docs/laya.md."""
        return (self.kind, self.n_options)

    def to_laya_public(self) -> dict[str, Any]:
        """The `agent.predict` format."""
        q: dict[str, Any] = {"type": self.kind, "instructions": self.instructions}
        if self.options is not None:
            q["criteria"] = self.options
        return q

    def to_laya_internal(self) -> dict[str, Any]:
        """The `build_sequence` / `render_options` format. NOT interchangeable."""
        q: dict[str, Any] = {"t": self.kind, "ins": self.instructions}
        if self.options is not None:
            q["crit"] = self.options
        return q


class Answer(BaseModel):
    """A decoded typed decision. The full distribution always crosses this boundary."""

    question_id: str
    kind: QuestionKind
    probs: list[float]
    top_index: int
    #: Laya's `answer_confidence` (= max p) — never its `confidence` field, which is
    #: uncalibrated normalized entropy. See docs/laya.md.
    answer_confidence: float
    expected_value: float | None = None
    temperature: float = 1.0


# ---------------------------------------------------------------------------
# Macro bins
#
# Bin edges are NOT defined here on purpose. They must be fitted from the
# training split and then frozen (configs/target_bins.yaml): a plausible-looking guess
# would quietly become the baseline, and refitting later silently invalidates
# every earlier number. `load_macro_bins` refuses to invent them.
# ---------------------------------------------------------------------------


class MacroBins(BaseModel):
    """Frozen, log-spaced bin edges for one macro."""

    macro: str
    edges: list[float] = Field(min_length=3)  # n_bins + 1, so at least 2 bins
    unit: str
    #: Bins are equal-width in log(value + offset). The offset sets where resolution goes:
    #: log1p (offset 1) wastes half the kcal bins below 10 kcal, where there is no data.
    offset: float = 1.0
    #: Provenance, so these can never be mistaken for a guess. e.g.
    #: "nutrition5k train split, n=3264, log-spaced quantiles, 2026-10-02"
    fitted_on: str

    @model_validator(mode="after")
    def _check_monotonic(self) -> MacroBins:
        if any(b <= a for a, b in pairwise(self.edges)):
            raise ValueError(f"{self.macro}: edges must be strictly increasing")
        if self.edges[0] < 0:
            raise ValueError(f"{self.macro}: edges must be non-negative")
        return self

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    def centres(self) -> list[float]:
        """Representative value per bin, used for the expected value.

        Geometric mean, to match the log spacing — an arithmetic centre would bias
        every estimate upward within its bin.
        """
        out = []
        for lo, hi in pairwise(self.edges):
            out.append((lo * hi) ** 0.5 if lo > 0 else hi / 2.0)
        return out

    def level_labels(self) -> list[str]:
        labels = []
        for i, (lo, hi) in enumerate(pairwise(self.edges)):
            last = i == self.n_bins - 1
            labels.append(f"{lo:g}+ {self.unit}" if last else f"{lo:g}-{hi:g} {self.unit}")
        return labels

    def index_of(self, value: float) -> int:
        """Bin containing `value`; clamped at both ends."""
        for i, hi in enumerate(self.edges[1:]):
            if value < hi:
                return i
        return self.n_bins - 1

    def to_question(self, instructions: str) -> Question:
        return Question(
            id=self.macro,
            kind="score",
            instructions=instructions,
            options=self.level_labels(),
        )


def load_macro_bins(path: str | Path, required: tuple[str, ...] = MACROS) -> dict[str, MacroBins]:
    """Load frozen bin definitions. Raises rather than defaulting.

    There is deliberately no fallback: see the module comment above. Pass
    `required=TARGETS` for bin files that also define mass, like configs/target_bins.yaml.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"No macro bin definition at {p}. Bins must be fitted from the training "
            "split and frozen before training (docs/datasets.md) — they are not "
            "guessable, and a guess would silently become the baseline."
        )
    raw = yaml.safe_load(p.read_text()) or {}
    bins = {m: MacroBins(macro=m, **spec) for m, spec in raw.items()}
    missing = set(required) - set(bins)
    if missing:
        raise ValueError(f"{p}: missing bin definitions for {sorted(missing)}")
    return bins


# ---------------------------------------------------------------------------
# Auxiliary questions
#
# Off the macro path: visual judgments meant to shape the representation and make the
# model inspectable. Defined for experiments; the released trainers do not use them.
# ---------------------------------------------------------------------------

FRIED = Question(
    id="fried",
    kind="noul",
    instructions="Does this meal contain a deep-fried or breaded item?",
)

SAUCED = Question(
    id="sauced",
    kind="noul",
    instructions="Is a substantial part of this meal covered in sauce, gravy or dressing?",
)

OIL_LEVEL = Question(
    id="oil_level",
    kind="score",
    instructions="How much visible oil or grease does this meal have?",
    options=["none", "a little", "moderate", "a lot", "very greasy"],
)

FOOD_CATEGORY = Question(
    id="food_category",
    kind="choice",
    instructions="What is the dominant food type in this meal?",
    options={
        "grain": "rice, noodles, bread, other starches",
        "meat": "pork, beef, chicken or other meat",
        "seafood": "fish, shrimp, shellfish",
        "egg": "egg-based dishes",
        "bean": "tofu, beans, other legumes",
        "vegetable": "vegetables or salad",
        "fruit": "fruit",
        "soup": "soup or broth",
        "dessert": "sweets, cake, ice cream",
        "mixed": "no single dominant type",
    },
)

#: Judgments that a text description of the meal alone cannot express.
THESIS_QUESTIONS: tuple[Question, ...] = (FRIED, SAUCED, OIL_LEVEL)

AUX_QUESTIONS: tuple[Question, ...] = (FRIED, SAUCED, OIL_LEVEL, FOOD_CATEGORY)
