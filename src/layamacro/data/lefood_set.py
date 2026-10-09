"""LeFood-Set (Mendeley cchsk79jkt, 2025): benchmark-only, never trained on.

524 Indonesian before/after photo pairs, CC BY 4.0. `data_original.xlsx` has one row per
**weighed** food item (digital scale): food name, before-eating photo and grams, after-eating
photo and grams. Verified 2026-10-01: 678 item rows, 50 distinct food names.

- Photo mass = sum of the rows that share the photo (a photo may hold several items).
- **Both** photos are samples: before-eating plates, and the weighed leftovers after eating,
  which are the partially eaten, messy plates the training datasets lack.
- Macros are **derived** (weighed grams x configs/nutrition_mapping/lefood_tkpi.csv) for a
  photo when its every item has an accepted row. *All* drafted rows were accepted,
  low-confidence ones included, because the derived macros are used **for benchmarking
  only**; photos relying on a low-confidence row are tagged `lowconf` so results can be
  split on it.
- The observer's 1-7 "visual estimation" column is an estimate and is never used: only
  weighed grams are ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path("data/raw/lefood_set")
MAPPING = Path("configs/nutrition_mapping/lefood_tkpi.csv")
_COLS = {
    "kcal": "kcal_100g",
    "protein_g": "protein_100g",
    "carbs_g": "carbs_100g",
    "fat_g": "fat_100g",
}


@dataclass(frozen=True)
class Photo:
    id: str
    image: Path
    mass_g: float
    stage: str  # "before" | "after"
    pair: str  # before-photo stem: before and after of one plate share it
    macros: dict[str, float] | None = None  # derived; None = some item unmapped
    lowconf: bool = False  # derived macros rely on a low-confidence mapping row


def mapping() -> dict[str, tuple[dict[str, float], bool]]:
    """food name -> (per-gram macros, low confidence?) for accepted rows only."""
    if not MAPPING.exists():
        return {}
    m = pd.read_csv(MAPPING, dtype=str, keep_default_na=False)
    ok = m[(m["use_for_macros"] == "yes") & (m["reviewed_by"].str.strip() != "")]
    return {
        r["food_name"]: (
            {t: float(r[c]) / 100.0 for t, c in _COLS.items()},
            r["confidence"] == "low",
        )
        for _, r in ok.iterrows()
    }


def load_photos(root: Path = ROOT) -> list[Photo]:
    df = pd.read_excel(root / "data_original.xlsx")
    table = mapping()
    out: list[Photo] = []
    for stage, img_col, w_col in (
        ("before", "Image Before Eaten", "Weight Before Eaten (g)"),
        ("after", "Image After Eaten", "Weight After Eaten (g)"),
    ):
        grams = df.groupby(img_col)[w_col].sum()
        pair = df.groupby(img_col)["Image Before Eaten"].first()
        food_col = "Name of the food"
        for name, g in grams.items():
            path = root / str(name)
            rows = df[df[img_col] == name]
            macros: dict[str, float] | None = {}
            low = False
            for _, r in rows.iterrows():
                hit = table.get(str(r[food_col]))
                if hit is None or macros is None:
                    macros = None
                    continue
                per_g, is_low = hit
                low = low or is_low
                for t, v in per_g.items():
                    macros[t] = macros.get(t, 0.0) + v * float(r[w_col])
            if g > 0 and path.exists():
                out.append(
                    Photo(
                        f"lefood_{Path(str(name)).stem}",
                        path,
                        float(g),
                        stage,
                        Path(str(pair[name])).stem,
                        macros or None,
                        low and bool(macros),
                    )
                )
    return out
