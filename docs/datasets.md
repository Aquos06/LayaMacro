# Datasets

LayaMacro trains only on **weighed** meals: every gram was measured on a scale, and calories
and macros come from those grams times a food-composition table. No label is a person's or a
model's visual estimate. Nothing here redistributes data: each script downloads from the
official source into `data/raw/` (gitignored).

## Training and evaluation

Each dataset keeps its own train / calibration / test split, and test results are always
reported per dataset. Calibration (temperature fitting) uses its own split, never test.

| dataset | photos | split by | train / calib / test | licence |
|---|---|---|---|---|
| **Nutrition5k** (cafeteria dishes, overhead camera) | 3,242 | dish | 2,466 / 274 / 502 | CC BY 4.0 |
| **DFood-TJ-F-Nutr** (Chinese hospital trays, phone) | 3,280 | patient | 2,181 / 407 / 692 | CC0 1.0 |
| **NutritionVerse-Real** (phone photos, random angles) | 889 | dish | 560 / 64 / 265 | CC BY-NC-SA 4.0 |

```bash
uv run python scripts/data/download_nutrition5k.py      # ~1.3 GB, overhead RGB only
uv run python scripts/data/download_dfood_tj.py         # 4.2 GB, no login needed
uv run python scripts/data/download_nutritionverse.py   # 1.1 GB, needs a Kaggle API token
```

### Nutrition5k

Thames et al., CVPR 2021, https://github.com/google-research-datasets/Nutrition5k. We use
the overhead RGB photo of each dish with the official depth-camera split, minus known bad
dishes (a photo of a smartphone, a ×1000 mass error, dishes whose every ingredient is named
`deprecated`, empty plates; listed in `layamacro.data.nutrition5k`). Calibration dishes are
carved from the official train split.

- **Never-seen meals** (`test_novel`, 256 dishes): test dishes whose exact ingredient
  combination never appears in training. 12 % of the official test set duplicates a training
  dish's exact ingredients and portions (often the same plate rescanned minutes apart), so
  results are reported on this subset too.

### DFood-TJ-F-Nutr

"Chinese Food Images for Full-cycle Nutrition Analysis Towards Diabetes Management",
Scientific Data 2026, dataset DOI https://doi.org/10.57760/sciencedb.28012. Meals of
hospitalised diabetic patients, photographed and weighed before eating, verified by the
hospital's nutrition department. Split **by patient** (no patient in two splits). The labels
are summed from the per-item rows of `truth_raw.xlsx`.

**These are patient photos.** No image from this dataset appears anywhere in this repository,
its documentation or its released files, and the benchmark scripts never print them.

### NutritionVerse-Real

arXiv:2309.07704, https://www.kaggle.com/datasets/nutritionverse/nutritionverse-real. 251
dishes in the metadata, 225 with photos (~4 per dish, random angles). The official split is
by dish with only Train / Val: Val is our test set, and calibration dishes are carved from
Train.

## Bins

Each target (kcal, protein, carbohydrate, fat, food weight) is predicted as a distribution
over 18 bins spaced evenly in log(value + offset), fitted once on the pooled training split and
frozen in `configs/target_bins.yaml`. Changing them invalidates every reported number.

## Benchmark only (never used for training)

| dataset | photos | what is weighed | licence |
|---|---|---|---|
| **Food Portion Benchmark** (ISSAI, IEEE Access 2025; Central Asian plates, lab) | 11,101 (authors' train + val) | grams per item | CC BY-NC 4.0 |
| **LeFood-Set** (Mendeley cchsk79jkt, 2025; Indonesian hospital plates) | 841 (524 before eating, 317 leftovers) | grams per item | CC BY 4.0 |

```bash
uv run python scripts/data/download_benchmark_datasets.py fpb lefood
```

No LayaMacro model trains on these, so they test how a model handles photos unlike its
training data. **Grams are the primary metric** (weighed). Calories and macros are
**derived** (weighed grams × a composition-table row per food,
`configs/nutrition_mapping/`) and reported in a separate table:

- FPB: USDA FoodData Central rows, 6,296 of 11,101 photos covered (every item on the plate
  must be mapped).
- LeFood-Set: Indonesian TKPI or USDA rows for all 50 foods; 178 photos rely on a
  low-confidence row (for example bone-in chicken weighed with the bone) and are also
  reported excluded.

FPB shows each plate from several angles, so its photos are not independent.
