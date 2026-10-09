"""NutritionVerse-Real: 889 phone photos of 225 weighed dishes. See docs/datasets.md.

Kaggle `nutritionverse/nutritionverse-real` (arXiv 2401.08598). Verified 2026-09-29:

- Labels: `nutritionverse_dish_metadata3.csv`, one row per dish — `total_food_weight` (g),
  `total_calories`, `total_protein`, `total_carbohydrates`, `total_fats`, plus minerals and
  per-item columns. Ingredients were weighed on a food scale; values come from packaging or
  the Canada Nutrient File. Labels pass the Atwater check (median gap 1.7 %).
- The metadata lists 251 dishes but **only 225 have photos** in the Kaggle release (the
  paper's "251 dishes" counts the metadata). Only photographed dishes become samples.
- Images: ~4 photos per dish from random angles (iPhone 13 Pro Max), file names begin
  `dish_<id>_`. The official split (`updated-manual-dataset-splits.csv`) is **by dish** with
  no dish on both sides, but has only Train / Val. Val becomes our **test**; calibration is
  carved from Train by dish.
- Much bigger meals than Nutrition5k: median 546 kcal, max 3,724 kcal / 391 g protein.
  Bins fitted on Nutrition5k alone miss 13 % of dishes, so the bins are fitted on all three
  training datasets.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path("data/raw/nutritionverse_real")
IMAGE_DIR = Path("nutritionverse-manual/nutritionverse-manual/images")
SPLITS_CSV = Path("nutritionverse-manual/nutritionverse-manual/updated-manual-dataset-splits.csv")

CALIBRATION_SEED = 20260929
CALIBRATION_FRACTION = 0.10  # of Train dishes

_DISH_RE = re.compile(r"^dish_(\d+)_")


@dataclass(frozen=True)
class Photo:
    id: str  # "nvr_" + file stem
    file_name: str
    dish: int
    official_split: str  # "Train" | "Val"
    kcal: float
    mass_g: float
    protein_g: float
    carbs_g: float
    fat_g: float

    def target(self, name: str) -> float:
        return float(getattr(self, name))


def load_photos(root: Path = DEFAULT_ROOT) -> dict[str, Photo]:
    dishes = pd.read_csv(root / "nutritionverse_dish_metadata3.csv").set_index("dish_id")
    splits = pd.read_csv(root / SPLITS_CSV)
    out: dict[str, Photo] = {}
    for fname, split in zip(splits["file_name"], splits["category"], strict=True):
        m = _DISH_RE.match(fname)
        if m is None:
            raise ValueError(f"unexpected image name {fname!r}")
        d = dishes.loc[int(m.group(1))]
        pid = "nvr_" + Path(fname).stem
        out[pid] = Photo(
            id=pid,
            file_name=fname,
            dish=int(m.group(1)),
            official_split=str(split),
            kcal=float(d["total_calories"]),
            mass_g=float(d["total_food_weight"]),
            protein_g=float(d["total_protein"]),
            carbs_g=float(d["total_carbohydrates"]),
            fat_g=float(d["total_fats"]),
        )
    return out


def image_path(photo: Photo, root: Path = DEFAULT_ROOT) -> Path:
    return root / IMAGE_DIR / photo.file_name


def dish_splits(photos: dict[str, Photo]) -> dict[str, str]:
    """photo id → split. Official Val → test; ~10 % of Train dishes → calibration."""
    train_dishes = sorted({p.dish for p in photos.values() if p.official_split == "Train"})
    rng = random.Random(CALIBRATION_SEED)
    calib = set(rng.sample(train_dishes, round(len(train_dishes) * CALIBRATION_FRACTION)))
    out = {}
    for p in photos.values():
        if p.official_split == "Val":
            out[p.id] = "test"
        elif p.official_split == "Train":
            out[p.id] = "calibration" if p.dish in calib else "train"
        else:
            raise ValueError(f"{p.id}: unknown official split {p.official_split!r}")
    return out
