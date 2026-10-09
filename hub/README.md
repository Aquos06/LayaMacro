---
license: cc-by-nc-sa-4.0
language: en
tags:
  - nutrition
  - food
  - calorie-estimation
  - uncertainty
  - laya
  - siglip2
base_model:
  - convaiinnovations/laya-typed-decisions
  - google/siglip2-large-patch16-384
---

# LayaMacro

**One photo of a meal → calories, protein, carbohydrate, fat and food weight, each with an
80 % uncertainty range.** Code, training and benchmarks: [github.com/Aquos06/LayaMacro](https://github.com/Aquos06/LayaMacro).

> **Research artifact, not a medical device.** Do not use it to make health or dietary
> decisions. Typical calorie error is 40-60 kcal per meal on the kinds of photos it was
> trained on, and much larger on unfamiliar ones.

## Files

| file | model | size | Nutrition5k kcal MAE |
|---|---|---|---|
| `layamacro-macroposterior.pt` | MacroPosterior: SigLIP 2 (last 12 blocks fine-tuned) + attention-pool head | 587 MB | 41.1 |
| `layamacro-laya.pt` | Laya graft: SigLIP 2 (last 4 blocks fine-tuned) -> projector -> Laya's decision head | 300 MB | 44.0 |

Each file holds only the trained weights, the calibration temperatures and the bin edges.
SigLIP 2 and Laya are downloaded from their official repositories on first use.

## Use

```bash
git clone https://github.com/Aquos06/LayaMacro && cd LayaMacro && uv sync
uv run python scripts/infer/predict.py \
    --weights hf://Aquos06/LayaMacro/layamacro-macroposterior.pt meal.jpg
```

```python
from layamacro.inference import load_predictor

model = load_predictor("hf://Aquos06/LayaMacro/layamacro-laya.pt")
model.predict(["meal.jpg"])
# [{"kcal": {"value": 339.9, "low": 230.0, "high": 460.0}, "protein_g": {...}, ...}]
```

The value is the mean of the model's probability distribution over 18 ranges per nutrient;
`low` and `high` bound its central 80 %.

## How it works

Both models read the photo with [SigLIP 2](https://huggingface.co/google/siglip2-large-patch16-384)
(576 patch tokens). **MacroPosterior** pools them with one learned query per nutrient and
predicts a distribution over 18 log-spaced ranges. The **Laya graft** projects the tokens
into [Laya](https://huggingface.co/convaiinnovations/laya), an encoder-only typed decision
model, after the question and its options, and reads one score per option from Laya's
`[MASK]` markers: no text is generated. Details in the GitHub repository
(`docs/architecture.md`).

## Training data

Only **weighed** meals (every gram measured; nutrients from grams × composition tables):

| dataset | photos (train / calibration / test) | licence |
|---|---|---|
| Nutrition5k (Thames et al., CVPR 2021) | 2,466 / 274 / 502 | CC BY 4.0 |
| DFood-TJ-F-Nutr (Scientific Data 2026, doi:10.57760/sciencedb.28012) | 2,181 / 407 / 692 | CC0 1.0 |
| NutritionVerse-Real (arXiv:2309.07704) | 560 / 64 / 265 | CC BY-NC-SA 4.0 |

Splits are by dish (Nutrition5k, NutritionVerse-Real) or by patient (DFood-TJ).
Calibration temperatures were fitted on the calibration split, never on test.

## Results (kcal mean absolute error, lower is better)

| | Nutrition5k | never-seen meals | DFood-TJ | NutritionVerse-Real |
|---|---|---|---|---|
| `layamacro-macroposterior` | 41.1 | 56.5 | 56.0 | 137.6 |
| `layamacro-laya` | 44.0 | 57.4 | 57.8 | 113.1 |
| OmniFood8K (CVPR 2026, run by us) | 36.8 | 50.5 | 168.9 | 639.3 |

On these test sets the 80 % ranges contain the true value 83 % of the time. OmniFood8K was
trained on Nutrition5k only, so its last two columns are transfer to new kinds of photos.
Full results, including negative ones: `docs/results.md` in the GitHub repository.

## Limitations

- **Portion size is guessed from one photo.** On photo types unlike the training data
  (tested on two datasets no model trained on), the model predicts the portion sizes it saw in
  training, and its 80 % ranges contain the truth only 9-25 % of the time.
- Trained on US cafeteria dishes, Chinese hospital meals and phone photos of mostly Western
  food; other cuisines are unfamiliar.
- The Laya checkpoint can be *asked* other questions (`model.ask`), but it was trained on
  nutrition only: on food-identity questions it answers at chance level.

## Licence and credits

Weights: **CC BY-NC-SA 4.0** (non-commercial; NutritionVerse-Real's licence carries over).
Code: Apache-2.0. Built on **Laya** by convaiinnovations (Apache-2.0) and **SigLIP 2** by
Google (Apache-2.0). LayaMacro is **not affiliated with or endorsed by** their authors or the
authors of the datasets above. There is no Laya paper; cite its model card and repository.
