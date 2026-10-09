"""Nutrition5k: metadata parser, exclusions, splits. See docs/datasets.md.

Traps this module exists to contain, each verified against the release:

- The dish CSVs have **no header and ragged rows**: 6 fixed fields + 7 per ingredient, macro
  order **fat, carb, protein**. The README's layout matches 0 of 5,006 rows, and
  `pandas.read_csv` truncates silently. Hence our own parser.
- Cafe 2 has every macro set to 0.0 — never load it for macro targets.
- Use the **depth** splits (the RealSense overhead subset), never the rgb ones: those are
  for side-angle video frames, and many of their dishes have no overhead image.
- Known-bad dishes are excluded **with a reason each**.
- 12 % of depth_test duplicates a training dish's exact ingredient+portion signature, so a
  leakage-free test subset is reported alongside the full one.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("data/raw/nutrition5k")

#: Ingredient ids literally named "deprecated".
DEPRECATED_INGREDIENTS = frozenset({"ingr_0000000453", "ingr_0000000458"})

#: Hand-verified exclusions. Rule-derived ones (all-deprecated dishes) are added in
#: `exclusions()` so the rule, not a pasted id list, is what's recorded.
MANUAL_EXCLUSIONS: dict[str, str] = {
    "dish_1559172356": "photo of a smartphone on the rig, not food (upstream issue #10)",
    "dish_1560974769": "x1000 mass error: 3,943 kcal / 844 g carb (upstream issue #11)",
    "dish_1557861216": "empty plate: 0 kcal, 'plate only'",
    "dish_1556575700": "empty plate",
}

#: Fraction of depth_train carved off for temperature fitting, and its seed. Fixed once;
#: changing either re-deals the calibration split and invalidates earlier ECE numbers.
CALIBRATION_FRACTION = 0.10
CALIBRATION_SEED = 20260928


@dataclass(frozen=True)
class Ingredient:
    id: str
    name: str
    grams: float
    kcal: float
    fat_g: float
    carbs_g: float
    protein_g: float


@dataclass(frozen=True)
class Dish:
    id: str
    kcal: float
    mass_g: float
    fat_g: float
    carbs_g: float
    protein_g: float
    ingredients: tuple[Ingredient, ...]

    def target(self, macro: str) -> float:
        return float(getattr(self, macro))

    @property
    def signature(self) -> tuple[tuple[str, float], ...]:
        """Exact ingredient+portion identity, used to detect rescans across splits."""
        return tuple(sorted((i.id, round(i.grams, 3)) for i in self.ingredients))

    @property
    def ingredient_set(self) -> frozenset[str]:
        """Which ingredients, ignoring portions: the "have we seen this meal?" key."""
        return frozenset(i.id for i in self.ingredients)

    @property
    def atwater_kcal(self) -> float:
        return 4 * self.protein_g + 4 * self.carbs_g + 9 * self.fat_g


def parse_dish_row(row: list[str]) -> Dish:
    """One ragged CSV row → Dish. Raises on any row that doesn't fit the real layout."""
    row = [c for c in row]
    while row and row[-1] == "":  # tolerate a trailing comma
        row.pop()
    if len(row) < 6 or (len(row) - 6) % 7:
        raise ValueError(f"bad row length {len(row)} (want 6 + 7n): {row[:3]}")
    dish_id, kcal, mass, fat, carb, protein = row[:6]
    ingrs = []
    for k in range(6, len(row), 7):
        iid, name, grams, ikcal, ifat, icarb, iprot = row[k : k + 7]
        if not iid.startswith("ingr_"):
            raise ValueError(f"{dish_id}: expected ingredient id at field {k}, got {iid!r}")
        ingrs.append(
            Ingredient(
                iid, name, float(grams), float(ikcal), float(ifat), float(icarb), float(iprot)
            )
        )
    return Dish(
        id=dish_id,
        kcal=float(kcal),
        mass_g=float(mass),
        fat_g=float(fat),
        carbs_g=float(carb),
        protein_g=float(protein),
        ingredients=tuple(ingrs),
    )


def load_dishes(root: Path = DEFAULT_ROOT) -> dict[str, Dish]:
    """Cafe 1 only. Cafe 2's macros are all zero."""
    path = root / "metadata" / "dish_metadata_cafe1.csv"
    with open(path, newline="") as f:
        dishes = [parse_dish_row(r) for r in csv.reader(f) if r]
    out = {d.id: d for d in dishes}
    if len(out) != len(dishes):
        raise ValueError(f"{path}: duplicate dish ids")
    return out


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def image_path(dish_id: str, root: Path = DEFAULT_ROOT) -> Path:
    return root / "imagery" / "realsense_overhead" / dish_id / "rgb.png"


def exclusions(dishes: dict[str, Dish]) -> dict[str, str]:
    """dish id → reason. Manual list plus every dish whose ingredients are all deprecated."""
    out = dict(MANUAL_EXCLUSIONS)
    for d in dishes.values():
        if d.ingredients and all(i.id in DEPRECATED_INGREDIENTS for i in d.ingredients):
            out.setdefault(d.id, "every ingredient is 'deprecated': no food identity in label")
    return out


@dataclass(frozen=True)
class Splits:
    train: tuple[str, ...]
    calibration: tuple[str, ...]
    test: tuple[str, ...]
    #: test ids with no exact signature match in train + calibration.
    test_clean: tuple[str, ...]
    #: test ids whose ingredient-id *set* never appears in train + calibration. Stricter than
    #: test_clean, and the closest Nutrition5k gets to an unseen meal.
    test_novel: tuple[str, ...]
    #: every id dropped, and why — including "no overhead image".
    dropped: dict[str, str]


def _carve_calibration(ids: list[str], dishes: dict[str, Dish]) -> tuple[list[str], list[str]]:
    """Hold out ~CALIBRATION_FRACTION of train, grouped by signature.

    Grouped so a rescanned plate never sits on both sides — the same leakage that
    affects test would otherwise flatter the calibration fit.
    """
    groups: dict[tuple[tuple[str, float], ...], list[str]] = {}
    for i in ids:
        groups.setdefault(dishes[i].signature, []).append(i)
    keys = sorted(groups)
    random.Random(CALIBRATION_SEED).shuffle(keys)
    target = round(len(ids) * CALIBRATION_FRACTION)
    calib: list[str] = []
    for k in keys:
        if len(calib) >= target:
            break
        calib += groups[k]
    cal = set(calib)
    return [i for i in ids if i not in cal], sorted(calib)


def build_splits(root: Path = DEFAULT_ROOT, dishes: dict[str, Dish] | None = None) -> Splits:
    dishes = dishes if dishes is not None else load_dishes(root)
    excl = exclusions(dishes)
    dropped: dict[str, str] = {}

    def usable(ids: Iterable[str]) -> list[str]:
        keep = []
        for i in ids:
            if i in excl:
                dropped[i] = excl[i]
            elif i not in dishes:
                dropped[i] = "not in cafe1 metadata"
            elif not image_path(i, root).exists():
                dropped[i] = "no overhead image"
            else:
                keep.append(i)
        return keep

    splits_dir = root / "dish_ids" / "splits"
    train_all = usable(read_ids(splits_dir / "depth_train_ids.txt"))
    test = usable(read_ids(splits_dir / "depth_test_ids.txt"))
    if set(train_all) & set(test):
        raise AssertionError("official depth splits overlap — the release has changed")

    train, calib = _carve_calibration(train_all, dishes)
    seen = {dishes[i].signature for i in train_all}
    test_clean = [i for i in test if dishes[i].signature not in seen]
    seen_sets = {dishes[i].ingredient_set for i in train_all}
    test_novel = [i for i in test if dishes[i].ingredient_set not in seen_sets]
    return Splits(
        tuple(train), tuple(calib), tuple(test), tuple(test_clean), tuple(test_novel), dropped
    )
