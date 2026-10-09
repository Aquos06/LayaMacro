"""The multi-task question bank.

Families, all typed decisions read from Laya's `[MASK]` markers (no new heads; Laya's
decision head is not modified):

- **N** nutrition: the 5 trained `score` questions (built by the trainer, not here).
- **I** ingredient: "Does this meal contain X?" as a 2-option `choice`, yes/no under neutral
  keys A/B in random order (upstream's workaround for `noul` following its option text).
- **D** which: "Which of these foods is on this plate?", 1 recorded item + 4 absent, A-E.
- **T** threshold: "Is this meal over 500 kcal?" and similar, from the weighed totals.

**Held out before any training** (fixed by `HOLDOUT_SEED`, never revised after results):
20 % of each dataset's food names are never asked about in training (neither as the answer
nor as a decoy); each family's last phrasing is used only at test; the T family is trained
on Nutrition5k only and tested on DFood-TJ and NutritionVerse-Real too.

The answer key is the recorded item lists (`item_labels.py`) and weighed totals: recorded,
never estimated.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from functools import cache

from layamacro.data.item_labels import INVISIBLE, item_names
from layamacro.data.samples import Sample
from layamacro.schema import Question

HOLDOUT_SEED = 20261007
HOLDOUT_FRACTION = 0.2
DATASETS = ("nutrition5k", "dfood_tj", "nutritionverse_real")
#: A name must occur on at least this many photos of its dataset to be asked about.
MIN_COUNT = 3

#: Phrasings per family; the LAST one is held out (test only).
ING_PHRASES = (
    "Does this meal contain {x}?",
    "Is there {x} in this food?",
    "Can {x} be found on this plate?",
    "Is {x} part of this meal?",
)
WHICH_PHRASES = (
    "Which of these foods is on this plate?",
    "Which one of these is part of the meal?",
    "One of these foods is in the photo. Which?",
    "Which item below appears in this meal?",
)
#: (target, threshold, phrasings); the last phrasing is held out.
THRESHOLDS = (
    (
        "kcal",
        500.0,
        (
            "Is this meal over 500 kcal?",
            "Does this food contain more than 500 kilocalories?",
            "Would this meal provide over 500 kcal of energy?",
            "Is the energy content of this meal above 500 kcal?",
        ),
    ),
    (
        "protein_g",
        30.0,
        (
            "Does this meal have more than 30 g of protein?",
            "Is there over 30 grams of protein in this food?",
            "Would this meal give more than 30 g of protein?",
            "Is the protein in this meal above 30 g?",
        ),
    ),
    (
        "carbs_g",
        60.0,
        (
            "Does this meal have more than 60 g of carbohydrate?",
            "Is there over 60 grams of carbohydrate in this food?",
            "Would this meal give more than 60 g of carbs?",
            "Is the carbohydrate in this meal above 60 g?",
        ),
    ),
    (
        "fat_g",
        20.0,
        (
            "Does this meal have more than 20 g of fat?",
            "Is there over 20 grams of fat in this food?",
            "Would this meal give more than 20 g of fat?",
            "Is the fat in this meal above 20 g?",
        ),
    ),
)
#: Datasets the T family trains on; the others are its held-out test.
T_TRAIN_DATASETS = ("nutrition5k",)


@dataclass(frozen=True)
class Asked:
    """One question about one photo, with its correct option key."""

    photo: str
    dataset: str
    family: str  # ingredient | which | threshold
    question: Question
    answer: str
    yes_key: str | None = None
    about: str = ""
    held_out: str = ""  # "" | "name" | "phrasing" | "dataset"
    invisible: bool = False


@cache
def vocab() -> dict[str, tuple[frozenset[str], frozenset[str]]]:
    """dataset -> (names asked in training, held-out names). Fixed by HOLDOUT_SEED.

    A name held out in any dataset is held out in all of them, so "never trained on" holds
    for the name itself, not just for one dataset's photos of it.
    """
    from layamacro.data.samples import load_samples

    names = item_names()
    pools, held_all = {}, set()
    for d in DATASETS:
        c = Counter(n for s in load_samples([d], check_images=False) for n in names.get(s.id, ()))
        pool = sorted(n for n, k in c.items() if k >= MIN_COUNT)
        rng = random.Random(f"{HOLDOUT_SEED}:{d}")
        held_all |= set(rng.sample(pool, round(HOLDOUT_FRACTION * len(pool))))
        pools[d] = frozenset(pool)
    held = frozenset(held_all)
    return {d: (pool - held, pool & held) for d, pool in pools.items()}


def _yes_no(text: str, truth: bool, rng: random.Random) -> tuple[Question, str, str]:
    yk, nk = ("A", "B") if rng.random() < 0.5 else ("B", "A")
    q = Question(
        id="yn",
        kind="choice",
        instructions=text,
        options=dict(sorted({yk: "yes", nk: "no"}.items())),
    )
    return q, (yk if truth else nk), yk


def ask(sample: Sample, rng: random.Random, train: bool, n_other: int = 4) -> list[Asked]:
    """Questions about one photo. train=True: only training names / phrasings / datasets,
    `n_other` sampled per call. train=False: one of every kind, held-out ones included and
    tagged."""
    names = item_names().get(sample.id)
    if names is None or sample.dataset not in DATASETS:
        return []
    known, held = vocab()[sample.dataset]
    phr = slice(0, -1) if train else slice(None)
    out: list[Asked] = []

    def pick_phrase(phrases: tuple[str, ...]) -> tuple[str, bool]:
        p = rng.choice(phrases[phr]) if train else rng.choice(phrases)
        return p, p == phrases[-1]

    def ingredient(name: str, present: bool, held_name: bool) -> Asked:
        text, held_phrase = pick_phrase(ING_PHRASES)
        q, a, yk = _yes_no(text.format(x=name), present, rng)
        tag = "name" if held_name else "phrasing" if held_phrase else ""
        return Asked(
            sample.id, sample.dataset, "ingredient", q, a, yk, name, tag, name in INVISIBLE
        )

    pos_known = sorted(names & known)
    neg_known = sorted(known - names)
    if train:
        kinds = ["ingredient+", "ingredient-", "which", "threshold"]
        while len(kinds) < n_other:
            kinds.append(rng.choice(["ingredient+", "ingredient-"]))
        for k in kinds[:n_other]:
            if k == "ingredient+" and pos_known:
                out.append(ingredient(rng.choice(pos_known), True, False))
            elif k == "ingredient-" and neg_known:
                out.append(ingredient(rng.choice(neg_known), False, False))
            elif k == "which" and pos_known and len(neg_known) >= 4:
                out.append(_which(sample, rng, pos_known, neg_known, False))
            elif k == "threshold" and sample.dataset in T_TRAIN_DATASETS:
                out.append(_threshold(sample, rng, train=True))
            elif neg_known:  # fill the slot so every photo gets n_other questions
                out.append(ingredient(rng.choice(neg_known), False, False))
        return out

    # test: 2 present + 2 absent known names, 1 present + 1 absent held-out name,
    # one "which" with known names, one with a held-out answer, all thresholds
    for n in rng.sample(pos_known, min(2, len(pos_known))):
        out.append(ingredient(n, True, False))
    for n in rng.sample(neg_known, min(2, len(neg_known))):
        out.append(ingredient(n, False, False))
    pos_held, neg_held = sorted(names & held), sorted(held - names)
    if pos_held:
        out.append(ingredient(rng.choice(pos_held), True, True))
    if neg_held:
        out.append(ingredient(rng.choice(neg_held), False, True))
    if pos_known and len(neg_known) >= 4:
        out.append(_which(sample, rng, pos_known, neg_known, False))
    if pos_held and len(neg_known) >= 4:
        out.append(_which(sample, rng, pos_held, neg_known, True))
    out += [_threshold(sample, rng, train=False, which=i) for i in range(len(THRESHOLDS))]
    return out


def _which(s: Sample, rng: random.Random, pos: list[str], neg: list[str], held_name: bool) -> Asked:
    true = rng.choice(pos)
    opts = [true, *rng.sample(neg, 4)]
    rng.shuffle(opts)
    keys = "ABCDE"
    text = rng.choice(WHICH_PHRASES[:-1])
    q = Question(
        id="which", kind="choice", instructions=text, options=dict(zip(keys, opts, strict=True))
    )
    return Asked(
        s.id,
        s.dataset,
        "which",
        q,
        keys[opts.index(true)],
        about=true,
        held_out="name" if held_name else "",
    )


def _threshold(s: Sample, rng: random.Random, train: bool, which: int | None = None) -> Asked:
    t, cut, phrases = THRESHOLDS[rng.randrange(len(THRESHOLDS)) if which is None else which]
    text = rng.choice(phrases[:-1]) if train else rng.choice(phrases)
    q, a, yk = _yes_no(text, s.targets[t] > cut, rng)
    tag = (
        "dataset" if s.dataset not in T_TRAIN_DATASETS
        else "phrasing" if text == phrases[-1]
        else ""
    )  # fmt: skip
    return Asked(s.id, s.dataset, "threshold", q, a, yk, f"{t}>{cut:g}", tag)
