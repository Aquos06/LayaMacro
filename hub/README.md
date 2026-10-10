---
license: cc-by-nc-sa-4.0
language: en
tags:
  - nutrition
  - food
  - calorie-estimation
  - food-recognition
  - uncertainty
  - laya
  - siglip2
base_model:
  - convaiinnovations/laya-typed-decisions
  - google/siglip2-large-patch16-384
---

# LayaMacro

**One photo of a meal → calories, protein, carbohydrate, fat and food weight, each with an
80 % uncertainty range, and answers to questions about the food.** Code, training and
benchmarks: [github.com/Aquos06/LayaMacro](https://github.com/Aquos06/LayaMacro).

> **Research artifact, not a medical device.** Do not use it to make health or dietary
> decisions. Average calorie error is 48-63 kcal per meal on Nutrition5k and DFood-TJ test
> photos, and much larger on unfamiliar kinds of photos.

## Files

| file | size | what |
|---|---|---|
| `layamacro-laya.pt` | 108 MB | the trained projector and Laya decision head, calibration temperatures, bin edges and the 243 trained food names |

SigLIP 2 and Laya themselves are downloaded from their official repositories on first use.

## Use

```bash
git clone https://github.com/Aquos06/LayaMacro && cd LayaMacro && uv sync
W=hf://Aquos06/LayaMacro/layamacro-laya.pt
uv run python scripts/infer/predict.py --weights $W meal.jpg --over kcal=500 \
    --ask "Does this meal contain rice?" --options yes no
```

```python
from layamacro.inference import load_predictor

model = load_predictor("hf://Aquos06/LayaMacro/layamacro-laya.pt")
model.predict(["meal.jpg"])        # [{"kcal": {"value": ..., "low": ..., "high": ...}, ...}]
model.prob_over(["meal.jpg"], "kcal", 500)                  # calibrated
model.ask(["meal.jpg"], "Which of these foods is on this plate?", ["rice", "noodles", "bread"])
model.foods                        # the 243 food names it was trained to recognise
```

## How it works

The photo goes through [SigLIP 2](https://huggingface.co/google/siglip2-large-patch16-384)
(frozen) into 576 image tokens. A projector places them into
[Laya](https://huggingface.co/convaiinnovations/laya), an encoder-only typed decision model,
**after** the question and its options. Laya scores every option at its `[MASK]` marker and
returns a probability per option: no text is generated. Nutrition is five questions over 18
value ranges each (estimate = mean, range = central 80 %); food questions are yes/no or
pick-one. Details: `docs/architecture.md` in the GitHub repository.

## Training data

Only **weighed** meals (every gram measured; nutrients from grams × composition tables). Food
names for the questions come from each dataset's recorded item lists.

| dataset | photos (train / calibration / test) | licence |
|---|---|---|
| Nutrition5k (Thames et al., CVPR 2021) | 2,466 / 274 / 502 | CC BY 4.0 |
| DFood-TJ-F-Nutr (Scientific Data 2026, doi:10.57760/sciencedb.28012) | 2,181 / 407 / 692 | CC0 1.0 |
| NutritionVerse-Real (arXiv:2309.07704) | 560 / 64 / 265 | CC BY-NC-SA 4.0 |

## Results

**Nutrition** (kcal mean absolute error, lower is better):

| | Nutrition5k | never-seen meals | DFood-TJ | NutritionVerse-Real |
|---|---|---|---|---|
| **LayaMacro** | 48.5 | 62.7 | 63.1 | 169.9 |
| OmniFood8K (CVPR 2026, run by us) | 36.8 | 50.5 | 168.9 | 639.3 |

On Nutrition5k the 80 % ranges contain the true value 84 % of the time. OmniFood8K was
trained on Nutrition5k only, so its last two columns are transfer to new kinds of photos.

**Food questions** (accuracy on test photos, three datasets pooled):

| question | chance | **LayaMacro** | Qwen3.5-2B, untrained |
|---|---|---|---|
| "Does this meal contain X?" (trained foods) | 0.56 | **0.89** | 0.80 |
| same, a wording never used in training | 0.55 | **0.88** | 0.79 |
| "Which of these 5 foods is on the plate?" | 0.20 | **0.86** | 0.62 |
| "Does this meal contain X?" (foods never trained on) | 0.63 | 0.65 | **0.83** |

Full results, including negative ones: `docs/results.md` in the GitHub repository.

## Limitations

- **Portion size is guessed from one photo.** On photo types unlike the training data, the
  model predicts the portion sizes it saw in training, and its 80 % ranges contain the truth
  only 13-23 % of the time.
- **Food recognition covers the 243 trained foods** (`model.foods`); for other foods it is
  near chance.
- **The training recipe is unstable**: another seed failed to learn the food questions; this
  is the seed that worked, verified on the benchmark.
- Trained on US cafeteria dishes, Chinese hospital meals and phone photos of mostly Western
  food; other cuisines are unfamiliar.

## Licence and credits

Weights: **CC BY-NC-SA 4.0** (non-commercial; NutritionVerse-Real's licence carries over).
Code: Apache-2.0. Built on **Laya** by convaiinnovations (Apache-2.0) and **SigLIP 2** by
Google (Apache-2.0). LayaMacro is **not affiliated with or endorsed by** their authors or the
authors of the datasets above. There is no Laya paper; cite its model card and repository.
