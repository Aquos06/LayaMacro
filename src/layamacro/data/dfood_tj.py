"""DFood-TJ-F-Nutr: Chinese hospital meals, weighed. See docs/datasets.md.

Scientific Data 2026, DOI 10.57760/sciencedb.28012, dataset licence CC0 1.0 (see NOTICE).
**Patient photos**: never commit or share an image.

Facts this module relies on, each verified against the release (2026-09-29):

- `truth/truth_raw.xlsx` has **one row per food item**. Only an image's first item row
  carries its `Image ID`; the following rows (NaN id) belong to the same image, so ids are
  forward-filled. Column names are **English**; the authors' own `Nutr.py` expects Chinese
  names and does not run against this release.
- Plate totals are sums over item rows. They match `truth/truth.xls` exactly for carbs, fat,
  protein and fiber; kcal disagrees for 3 of 3,280 images, where `truth.xls` is the one that
  is wrong (image 55: 149 kcal listed, 257 by items and by Atwater). Hence: sum the items.
- Mass = sum of item `Weight (g)`. Only 11 % of item weights are multiples of 10, which is
  consistent with real weighing rather than nominal portions.
- 91 patients, 958 distinct meals, ~3.4 photos per meal; 60 % of images share their exact
  label with another image (the same meal photographed again). So the split is **by
  patient** — the authors' first-80 %-of-rows split is not used.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path("data/raw/dfood_tj/DFoodTJ-F-Nutr")

#: Patient-level split, fixed once. Changing either re-deals every split.
SPLIT_SEED = 20260929
#: Target image fractions; whole patients are assigned greedily, so actuals differ slightly.
TEST_FRACTION = 0.20
CALIBRATION_FRACTION = 0.10


@dataclass(frozen=True)
class Meal:
    id: str  # "dftj_000001"
    image_number: int
    patient: int
    meal: str  # e.g. "2020/11/9Dinner"
    kcal: float
    mass_g: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    items: tuple[tuple[str, float], ...]  # (food name, grams)

    def target(self, name: str) -> float:
        return float(getattr(self, name))


def load_meals(root: Path = DEFAULT_ROOT) -> dict[str, Meal]:
    raw = pd.read_excel(root / "truth" / "truth_raw.xlsx", sheet_name="Sheet1")
    raw["img"] = raw["Image ID"].ffill().astype(int)
    if raw.groupby("img")["Patient ID"].nunique().max() != 1:
        raise ValueError("an image spans several patients — the release has changed")
    out: dict[str, Meal] = {}
    for img, rows in raw.groupby("img", sort=True):
        first = rows.iloc[0]
        mid = f"dftj_{int(img):06d}"
        out[mid] = Meal(
            id=mid,
            image_number=int(img),
            patient=int(first["Patient ID"]),
            meal=str(first["Date"]),
            kcal=float(rows["Calorie (kcal)"].sum()),
            mass_g=float(rows["Weight (g)"].sum()),
            protein_g=float(rows["Protein (g)"].sum()),
            carbs_g=float(rows["Carbohydrate (g)"].sum()),
            fat_g=float(rows["Fat (g)"].sum()),
            fiber_g=float(rows["Fiber (g)"].sum()),
            items=tuple(
                (str(n), float(w))
                for n, w in zip(rows["Food Name"], rows["Weight (g)"], strict=True)
            ),
        )
    return out


def image_path(meal: Meal, root: Path = DEFAULT_ROOT) -> Path:
    return root / "img" / f"{meal.image_number:06d}.png"


def patient_splits(meals: dict[str, Meal]) -> dict[str, str]:
    """meal id → 'train' | 'calibration' | 'test', whole patients at a time."""
    by_patient: dict[int, list[str]] = {}
    for m in meals.values():
        by_patient.setdefault(m.patient, []).append(m.id)
    patients = sorted(by_patient)
    random.Random(SPLIT_SEED).shuffle(patients)
    total = len(meals)
    quota = {
        "test": math.floor(total * TEST_FRACTION),
        "calibration": math.floor(total * CALIBRATION_FRACTION),
    }
    filled = {"test": 0, "calibration": 0}
    out: dict[str, str] = {}
    for p in patients:
        ids = by_patient[p]
        split = next((s for s in ("test", "calibration") if filled[s] < quota[s]), "train")
        if split != "train":
            filled[split] += len(ids)
        for i in ids:
            out[i] = split
    return out
