"""The unseen-dataset benchmark: datasets no model here is trained on.

Food Portion Benchmark (Central Asian lab plates, 11,101 photos of the authors' train + val
splits) and LeFood-Set (Indonesian hospital plates, 841 before/after photos). Both are
benchmark-only, never used for training, so every model of ours and every downloaded model
meets them for the first time: a zero-shot head-to-head.

- **Mass is weighed**: the primary metric here, since only weighed grams are ground truth.
- kcal / protein / carbs / fat are **derived** (weighed grams x composition-table row,
  configs/nutrition_mapping/) and only for photos whose every item is mapped. They are
  reported in a separate, labelled table, never mixed with weighed test numbers.
- FPB shows each plate from several angles, so its photos are not independent: treat its
  numbers as one dataset's, not 11k independent trials.
"""

from __future__ import annotations

from layamacro.data.samples import BENCH_DATASETS, Sample, load_samples


def bench_samples() -> list[Sample]:
    """Every benchmark photo, in a fixed order (dataset, then id)."""
    s = load_samples(BENCH_DATASETS, check_images=False)
    return sorted(s, key=lambda x: (x.dataset, x.id))
