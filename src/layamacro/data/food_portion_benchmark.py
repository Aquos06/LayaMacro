"""Food Portion Benchmark (ISSAI, IEEE Access 2025): benchmark-only, never trained on.

HF `issai/Food_Portion_Benchmark`, CC BY-NC 4.0. 138 Central Asian classes, lab photos (top
+ four side views, RealSense + phones). Labels are YOLO lines with a 6th column: the item's
**weighed** grams. Verified 2026-10-01 on the downloaded files.

- Plate mass = sum of the item grams (measured).
- Plate macros are **derived** only when *every* item's class has
  `use_for_macros = yes` and a `reviewed_by` entry in configs/nutrition_mapping/fpb_fdc.csv.
  Otherwise the macros are left out (unknown), never partially summed.
- Only the authors' train and val splits are used. Test labels are meant to stay hidden for
  their leaderboard, so we leave test alone.
- Each dish appears from several angles, so its photos are not independent: treat its
  benchmark numbers as one dataset's, not 11k independent trials.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from layamacro.schema import MACROS

ROOT = Path("data/raw/food_portion_benchmark/FPB_Dataset/RGB")
MAPPING = Path("configs/nutrition_mapping/fpb_fdc.csv")
SPLITS = ("train", "val")
_COLS = {
    "kcal": "kcal_100g",
    "protein_g": "protein_100g",
    "carbs_g": "carbs_100g",
    "fat_g": "fat_100g",
}


@dataclass(frozen=True)
class Plate:
    id: str
    image: Path
    targets: dict[str, float]
    derived: tuple[str, ...]
    n_items: int


def per_gram_table() -> dict[int, dict[str, float]]:
    """class id -> per-gram macros, for approved rows only."""
    m = pd.read_csv(MAPPING, dtype=str, keep_default_na=False)
    ok = m[(m["use_for_macros"] == "yes") & (m["reviewed_by"].str.strip() != "")]
    return {
        int(r["class_id"]): {t: float(r[c]) / 100.0 for t, c in _COLS.items()}
        for _, r in ok.iterrows()
    }


def load_plates(root: Path = ROOT) -> list[Plate]:
    table = per_gram_table()
    out: list[Plate] = []
    for split in SPLITS:
        for lbl in sorted((root / split / "labels").glob("*.txt")):
            img = root / split / "images" / (lbl.stem + ".jpg")
            if not img.exists():
                continue  # download still in progress, or an unmatched label
            items: list[tuple[int, float]] = []
            complete = True
            for line in lbl.read_text().split("\n"):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) < 6:
                    complete = False  # an item without a weight: plate mass unknown
                    break
                items.append((int(parts[0]), float(parts[5])))
            if not complete or not items or any(g <= 0 for _, g in items):
                continue
            targets = {"mass_g": sum(g for _, g in items)}
            derived: tuple[str, ...] = ()
            if all(c in table for c, _ in items):
                for t in MACROS:
                    targets[t] = sum(g * table[c][t] for c, g in items)
                derived = MACROS
            out.append(Plate(f"fpb_{split}_{lbl.stem}", img, targets, derived, len(items)))
    return out
