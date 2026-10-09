"""All datasets as one list of samples with the same shape.

Every model, ours and the baselines, trains and is scored on exactly these samples, so the
comparison is like for like. Each dataset keeps **its own** train / calibration / test, and
test is always reported per dataset — never pooled, since pooling lets the biggest dataset
hide failure on the others.

`group` is the leakage unit: a group never spans two splits (asserted in `load_samples`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from layamacro.data import dfood_tj, nutrition5k, nutritionverse_real
from layamacro.schema import TARGETS

SPLITS = ("train", "calibration", "test")
#: The evaluation datasets: own train / calibration / test, every target measured.
DATASETS = ("nutrition5k", "dfood_tj", "nutritionverse_real")
#: Benchmark-only datasets, never trained on: only the unseen-dataset benchmark
#: (eval/bench.py) loads them. They have no splits of their own, so every sample carries
#: split "train" as a placeholder, and only the targets that are known. Mass is always
#: weighed; macros are either derived from a reviewed table mapping or absent.
BENCH_DATASETS = ("food_portion_benchmark", "lefood_set")


@dataclass(frozen=True)
class Sample:
    dataset: str
    id: str
    image: Path
    split: str
    group: str  # leakage unit: dish (N5k), patient (DFood-TJ), dish (NV-Real)
    #: DATASETS: every name in TARGETS. BENCH_DATASETS: a subset that always includes mass_g;
    #: a missing key means "unknown", never zero.
    targets: dict[str, float]
    #: extra report subsets, e.g. "test_clean", "test_novel" (Nutrition5k)
    tags: tuple[str, ...] = field(default=())
    #: targets computed as weighed grams x a composition-table row, not reported directly
    derived: frozenset[str] = field(default=frozenset())


def _nutrition5k() -> list[Sample]:
    dishes = nutrition5k.load_dishes()
    s = nutrition5k.build_splits(dishes=dishes)
    clean, novel = set(s.test_clean), set(s.test_novel)
    out = []
    for split, ids in (("train", s.train), ("calibration", s.calibration), ("test", s.test)):
        for i in ids:
            tags = tuple(t for t, sub in (("test_clean", clean), ("test_novel", novel)) if i in sub)
            out.append(
                Sample(
                    dataset="nutrition5k",
                    id=i,
                    image=nutrition5k.image_path(i),
                    split=split,
                    # the calibration carve already groups by signature; the dish is the
                    # unit that must not repeat across splits
                    group=i,
                    targets={t: dishes[i].target(t) for t in TARGETS},
                    tags=tags,
                )
            )
    return out


def _dfood_tj() -> list[Sample]:
    meals = dfood_tj.load_meals()
    split = dfood_tj.patient_splits(meals)
    return [
        Sample(
            dataset="dfood_tj",
            id=m.id,
            image=dfood_tj.image_path(m),
            split=split[m.id],
            group=f"patient_{m.patient}",
            targets={t: m.target(t) for t in TARGETS},
        )
        for m in meals.values()
    ]


def _nutritionverse_real() -> list[Sample]:
    photos = nutritionverse_real.load_photos()
    split = nutritionverse_real.dish_splits(photos)
    return [
        Sample(
            dataset="nutritionverse_real",
            id=p.id,
            image=nutritionverse_real.image_path(p),
            split=split[p.id],
            group=f"dish_{p.dish}",
            targets={t: p.target(t) for t in TARGETS},
        )
        for p in photos.values()
    ]


def _food_portion_benchmark() -> list[Sample]:
    from layamacro.data import food_portion_benchmark as fpb

    return [
        Sample(
            dataset="food_portion_benchmark",
            id=p.id,
            image=p.image,
            split="train",
            group=p.id,
            targets=p.targets,
            derived=frozenset(p.derived),
        )
        for p in fpb.load_plates()
    ]


def _lefood_set() -> list[Sample]:
    from layamacro.data import lefood_set

    return [
        Sample(
            dataset="lefood_set",
            id=p.id,
            image=p.image,
            split="train",
            group=f"plate_{p.pair}",  # before and after photos of one plate stay together
            targets={"mass_g": p.mass_g, **(p.macros or {})},
            derived=frozenset(p.macros or ()),
            tags=(p.stage,) + (("lowconf",) if p.lowconf else ()),
        )
        for p in lefood_set.load_photos()
    ]


_LOADERS = {
    "nutrition5k": _nutrition5k,
    "dfood_tj": _dfood_tj,
    "nutritionverse_real": _nutritionverse_real,
    "food_portion_benchmark": _food_portion_benchmark,
    "lefood_set": _lefood_set,
}


def load_samples(datasets: Iterable[str] = DATASETS, check_images: bool = True) -> list[Sample]:
    """Load, then assert the invariants every downstream number depends on."""
    out: list[Sample] = []
    for d in datasets:
        out += _LOADERS[d]()
    ids = [s.id for s in out]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate sample ids across datasets")
    seen: dict[tuple[str, str], str] = {}
    for s in out:
        if s.split not in SPLITS:
            raise AssertionError(f"{s.id}: bad split {s.split!r}")
        prev = seen.setdefault((s.dataset, s.group), s.split)
        if prev != s.split:
            raise AssertionError(f"leakage: {s.dataset} group {s.group} in {prev} and {s.split}")
        if s.dataset in DATASETS and set(s.targets) != set(TARGETS):
            raise AssertionError(f"{s.id}: targets {sorted(s.targets)} != {sorted(TARGETS)}")
        if s.dataset not in DATASETS:
            if s.split != "train":
                raise AssertionError(f"{s.id}: benchmark data must have split 'train'")
            if "mass_g" not in s.targets or not set(s.targets) <= set(TARGETS):
                raise AssertionError(f"{s.id}: benchmark targets {sorted(s.targets)}")
            if not s.derived <= set(s.targets):
                raise AssertionError(f"{s.id}: derived {sorted(s.derived)} not in targets")
        if check_images and not s.image.exists():
            raise AssertionError(f"{s.id}: missing image {s.image}")
    return out
