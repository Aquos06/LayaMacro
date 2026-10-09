"""Which foods are on each photo: the recorded item lists, as clean names.

These are **recorded labels**, written when each meal was built and weighed, never an
estimate: Nutrition5k's per-dish ingredient rows, DFood-TJ's per-item truth rows,
NutritionVerse-Real's per-dish `food_item_type_k` columns. They answer "is X on the plate?"
and "which of these is on the plate?" for the multi-task question bank (`questions.py`)
and its benchmark (scripts/benchmark/question_bank.py).

Names are normalised once here (lowercase, NutritionVerse product slugs turned into words).
Nutrition5k lists seasonings and oils nobody can see in a photo; they are flagged
`INVISIBLE` so results can be split on them instead of hiding them.
"""

from __future__ import annotations

import re
from functools import cache

import pandas as pd

from layamacro.data import dfood_tj, nutrition5k, nutritionverse_real

#: Nutrition5k ingredients that are mixed in or poured on, so not visible as an item.
INVISIBLE = frozenset(
    {
        "olive oil", "salt", "garlic", "pepper", "vinegar", "lemon juice", "mustard",
        "soy sauce", "vinaigrette", "cream", "mayonnaise", "sugar", "wine", "butter",
        "white wine", "caesar dressing", "sour cream", "ginger", "orange juice", "thyme",
        "oregano", "rosemary", "honey", "canola oil", "vegetable oil", "sesame oil",
    }
)  # fmt: skip


def _nvr_name(slug: str) -> str:
    """'costco-shrimp-sushi-roll-1' -> 'shrimp sushi roll'."""
    s = re.sub(r"[-_]+", " ", slug.strip().lower())
    s = re.sub(r"\s+\d+$", "", s)
    s = re.sub(r"^(costco|trader joes?|walmart)\s+", "", s)
    return s.strip()


@cache
def item_names() -> dict[str, frozenset[str]]:
    """sample id -> the set of food names recorded for that photo (all three datasets)."""
    out: dict[str, frozenset[str]] = {}
    for dish_id, d in nutrition5k.load_dishes().items():
        names = {i.name.strip().lower() for i in d.ingredients if i.name != "deprecated"}
        if names:
            out[dish_id] = frozenset(names)
    for mid, m in dfood_tj.load_meals().items():
        out[mid] = frozenset(n.strip().lower() for n, _ in m.items)
    root = nutritionverse_real.DEFAULT_ROOT
    dishes = pd.read_csv(root / "nutritionverse_dish_metadata3.csv").set_index("dish_id")
    cols = [c for c in dishes.columns if c.startswith("food_item_type_")]
    per_dish = {
        int(i): frozenset(_nvr_name(str(v)) for v in row[cols].dropna())
        for i, row in dishes.iterrows()
    }
    for pid, p in nutritionverse_real.load_photos().items():
        if per_dish.get(p.dish):
            out[pid] = per_dish[p.dish]
    return out
