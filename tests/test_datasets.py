"""Facts about the three real datasets that every number downstream depends on.

Marked `data`: they need the downloads in data/raw/. If one fails, a release changed or a
split was re-dealt — fix docs/DATA.md and the loader together; never relax the assertion.
"""

from __future__ import annotations

import collections

import pytest

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def samples():  # type: ignore[no-untyped-def]
    from layamacro.data.samples import load_samples

    return load_samples()  # asserts no group spans two splits, all images exist


def _count(samples, dataset):  # type: ignore[no-untyped-def]
    return collections.Counter(s.split for s in samples if s.dataset == dataset)


def test_nutrition5k_counts_unchanged(samples) -> None:  # type: ignore[no-untyped-def]
    assert _count(samples, "nutrition5k") == {"train": 2466, "calibration": 274, "test": 502}
    tags = collections.Counter(t for s in samples for t in s.tags)
    assert tags == {"test_clean": 445, "test_novel": 256}


def test_dfood_tj_patient_split(samples) -> None:  # type: ignore[no-untyped-def]
    d = [s for s in samples if s.dataset == "dfood_tj"]
    assert len(d) == 3280
    assert len({s.group for s in d}) == 91
    assert _count(samples, "dfood_tj") == {"train": 2181, "calibration": 407, "test": 692}


def test_dfood_tj_sums_items_not_truth_xls() -> None:
    from layamacro.data.dfood_tj import load_meals

    m = load_meals()["dftj_000055"]
    # truth.xls lists 148.98 kcal here; the item rows sum to 257.44, and Atwater agrees.
    assert m.kcal == pytest.approx(257.44, abs=0.01)
    at = 4 * m.protein_g + 4 * m.carbs_g + 9 * m.fat_g
    assert abs(at - m.kcal) / m.kcal < 0.05


def test_nutritionverse_real_dish_split(samples) -> None:  # type: ignore[no-untyped-def]
    d = [s for s in samples if s.dataset == "nutritionverse_real"]
    assert len(d) == 889
    # 251 dishes in the metadata, but only 225 have photos in the Kaggle release
    assert len({s.group for s in d}) == 225
    assert _count(samples, "nutritionverse_real") == {"train": 560, "calibration": 64, "test": 265}


def test_bins_cover_every_training_value(samples) -> None:  # type: ignore[no-untyped-def]
    from layamacro.schema import TARGETS, load_macro_bins

    bins = load_macro_bins("configs/target_bins.yaml", required=TARGETS)
    for t in TARGETS:
        top = bins[t].edges[-1]
        assert all(s.targets[t] < top for s in samples if s.split == "train"), t
