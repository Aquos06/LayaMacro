# Results

Measured on the test splits in `docs/datasets.md` during the research behind LayaMacro
(October 2026). The scripts in `scripts/benchmark/` reproduce them from the released
checkpoint or from your own runs; asking the bank's questions through the released file
reproduces the benchmark (e.g. "contains X?" on Nutrition5k: 0.891 vs 0.892).

The released checkpoint is **Laya, multi-task, seed 0**: SigLIP 2 frozen, Laya's backbone
frozen, the projector and Laya's decision head trained for 10 epochs on nutrition and food
questions. Numbers marked "mean of N seeds" average N trainings of the same recipe; Laya varies
a lot between seeds.

## Metrics

- **kcal MAE**: mean absolute calorie error per meal (lower is better). **r**: correlation
  between predicted and true calories.
- **80 % range coverage**: how often the true value lies inside the model's 80 % range
  (should be ≈ 0.80).
- **ECE**: calibration error of the bin probabilities (lower is better).
- **Accuracy** on food questions, next to **chance**: always giving the most common answer
  (yes/no), or 0.20 for a 1-of-5 pick.

## 1. Nutrition

kcal MAE, lower is better.

| model | Nutrition5k | never-seen meals | DFood-TJ | NutritionVerse-Real |
|---|---|---|---|---|
| **LayaMacro, released checkpoint** | 48.5 (r 0.94) | 62.7 | 63.1 | 169.9 |
| LayaMacro recipe, mean of 2 seeds | 49.1 | 61.7 | 68.8 | 172.0 |
| same recipe without food questions, mean of 2 seeds | 46.0 | 59.1 | 70.2 | 137.3 |
| *no image (predict the training distribution)* | *200.2* | *158.6* | *193.4* | *565.1* |

The released checkpoint's 80 % ranges contain the truth 84 % of the time on Nutrition5k
(ECE 0.08), 92 % on DFood-TJ (too wide there) and 79 % on NutritionVerse-Real.

## 2. Against published models we downloaded and ran

Same test photos, same metrics. These models were trained by their authors on Nutrition5k
only, so DFood-TJ and NutritionVerse-Real measure transfer to a new kind of photo for them,
not a like-for-like comparison.

| model | Nutrition5k | never-seen | DFood-TJ | NV-Real |
|---|---|---|---|---|
| OmniFood8K method (CVPR 2026, official weights) | **36.8** | **50.5** | 168.9 | 639.3 |
| **LayaMacro** | 48.5 | 62.7 | **63.1** | **169.9** |
| FoodLMM (2023, official weights, 7B) | 48.6 | 66.2 | 209.3 | 684.7 |
| Food-R1 (2026, official weights, 8B; kcal only) | 89.7 | 116.5 | 116.3 | 447.6 |

- On Nutrition5k, **OmniFood8K is clearly better**. It fine-tunes two full backbones (287 M
  parameters) and adds a predicted depth map; its training script also keeps the best of 150
  epochs by test-set score, which flatters its number by roughly 2-3 kcal.
- Two downloadable models are **not ranked** because they were trained on Nutrition5k test
  dishes: third-party DPF-Nutrition weights (18.4 kcal, the same as on their training
  dishes) and CalorieCLIP (its card describes a random split of all dishes).

## 3. Photos unlike any training data (no model trained on these)

Food Portion Benchmark (Central Asian lab plates) and LeFood-Set (Indonesian hospital
plates). **Grams error (weighed)**, lower is better:

| model | FPB | LeFood, before eating | LeFood, leftovers | 80 % coverage |
|---|---|---|---|---|
| **LayaMacro** | **194** | **77** | 66 | 0.13-0.23 |
| OmniFood8K (CVPR 2026) | 283 | 78 | **61** | — |
| *no image* | *214* | *151* | *177* | — |

- **Every model is poor at portion size on unfamiliar photos.** FPB plates (median 230 g) are
  predicted far too small by every model, LeFood's single items (median 46 g) far too large:
  each model guesses the portion sizes of its own training data. A single photo carries no
  absolute scale.
- **Uncertainty ranges are over-confident here**: they contain the truth only 13-23 % of the
  time, against 79-92 % on the training datasets' test photos.

## 4. Food questions

Accuracy on test photos, pooled over Nutrition5k, DFood-TJ and NutritionVerse-Real
(threshold questions: Nutrition5k, where always answering "no" scores 0.84). In brackets: the
same question asked about a photo of a *different* meal; a real image-based answer drops there.

| question | chance | **LayaMacro** | Qwen3.5-2B, untrained | Gemma 4 E2B, untrained |
|---|---|---|---|---|
| "Does this meal contain X?", trained foods | 0.56 | **0.89** (0.64) | 0.80 | 0.74 |
| same, a wording never used in training | 0.55 | **0.88** (0.64) | 0.79 | 0.68 |
| same, food names never trained on | 0.63 | 0.65 (0.63) | **0.83** | 0.75 |
| "Which of these 5 foods is on the plate?", trained | 0.20 | **0.86** (0.41) | 0.62 | 0.52 |
| same, the right answer never trained on | 0.20 | 0.40 (0.32) | **0.55** | 0.51 |
| "Is this meal over 500 kcal?" and similar (Nutrition5k) | 0.84 | **0.92** (0.77) | 0.61 | 0.50 |
| same, on DFood-TJ and NV-Real (never trained there) | 0.67 | **0.70** (0.67) | 0.54 | 0.51 |

- **It recognises the foods it was trained on, from the image**: accuracy drops sharply when
  the photo is swapped for another meal's.
- **New wordings work** as well as trained ones.
- **New foods mostly don't**: untrained VLMs, with their broad pre-training, beat it there.
- **Threshold questions barely transfer** to datasets where they were never trained (0.70 vs
  0.67 for always answering "no"). The nutrition distribution (`prob_over`) is calibrated
  and preferable for them.

Per-dataset accuracy for the released checkpoint ("contains X?", trained foods): Nutrition5k
0.89, DFood-TJ 0.91, NutritionVerse-Real 0.85; "which of 5": 0.87 / 0.88 / 0.82.

## 5. What training adds

Untrained models, as downloaded, on Nutrition5k (kcal MAE): Gemma 4 E2B 310, Qwen3.5-2B 417,
Laya without an image 482, all worse than guessing without the photo (200). Training brings
LayaMacro to 48.5 kcal, and food-question accuracy from chance to 0.86-0.89.

## 6. Where the calorie error comes from

Calories = grams × calories per gram. For the nutrition-only Laya, replacing one predicted
factor with the truth shows its share: perfect grams cut calorie error by only 11-15 %,
perfect calories per gram by up to 60 % (DFood-TJ). **Recognising what is on the plate
matters more than portion size** on these datasets.

## Negative results, kept on purpose

- **Unstable training**: a second seed of the released recipe stayed near chance on the food
  questions (final loss 1.80 vs 1.35); the released checkpoint is the seed that learned them.
- **Food questions cost calorie accuracy** (section 1: 46.0 → 49.1 kcal, mean of 2 seeds).
- **Training SigLIP 2's last blocks inside Laya did not reliably help**: with 8 blocks the two
  seeds gave 43.4 and 50.6 kcal on Nutrition5k.
- **20 instead of 10 epochs** shifted which dataset it fits, not the average.
- **Over-confident ranges on unfamiliar photos** (section 3).
