# Results

All numbers were measured on the test splits in `docs/datasets.md`, with the metrics below,
during the research that produced LayaMacro (October 2026). The scripts in
`scripts/benchmark/` reproduce them from the released checkpoints or from your own runs.
"Mean of N seeds" means the same recipe trained N times with different random seeds; Laya in
particular varies a lot between seeds, so single-seed numbers are marked.

## Metrics

- **kcal MAE**: mean absolute calorie error per meal (lower is better). **r**: correlation
  between predicted and true calories.
- **80 % range coverage**: how often the true value lies inside the model's 80 % range. It
  should be ≈ 0.80: lower means over-confident, higher means ranges wider than needed.
- **ECE**: calibration error of the bin probabilities (lower is better).
- **F1 ±1**: macro F1 over the 18 bins, counting the neighbouring bin as correct, averaged
  over the five targets.

## Released checkpoints

| checkpoint | what | N5k kcal MAE | never-seen | DFood-TJ | NV-Real | 80 % coverage (N5k) |
|---|---|---|---|---|---|---|
| `layamacro-macroposterior` | MacroPosterior, last 12 SigLIP 2 blocks trained, seed 0 | **41.1** | 56.5 | **56.0** | 137.6 | 0.83 |
| `layamacro-laya` | Laya graft, last 4 SigLIP 2 blocks trained, seed 0 | 44.0 | 57.4 | 57.8 | **113.1** | 0.83 |

## 1. Test sets (mean of seeds; kcal MAE, lower is better)

| model | seeds | Nutrition5k | never-seen meals | DFood-TJ | NutritionVerse-Real |
|---|---|---|---|---|---|
| **MacroPosterior, 12 SigLIP 2 blocks trained** | 2 | **41.4** (r 0.95) | **56.7** | **57.0** | 130.3 |
| MacroPosterior, frozen encoder | 3 | 48.1 | 63.8 | 65.1 | 159.5 |
| **Laya graft, 4 SigLIP 2 blocks trained** | 2 | 45.7 (r 0.94) | 59.8 | 61.0 | **121.6** |
| Laya graft, frozen encoder | 2 | 46.0 | 59.1 | 70.2 | 137.3 |
| Laya graft, multi-task, frozen encoder | 2 | 49.1 | 61.7 | 68.8 | 172.0 |
| Gemma 4 E2B, fine-tuned (LoRA + head) | 3 | 49.6 | 65.3 | 85.9 | 185.0 |
| Qwen3.5-2B, fine-tuned (LoRA + head) | 3 | 56.9 | 70.4 | 78.7 | 210.6 |
| *no image (predict the training distribution)* | — | *200.2* | *158.6* | *193.4* | *565.1* |

Calibration holds on these datasets: 80 % ranges contain the truth 80-84 % of the time and
ECE is 0.07-0.09 for the MacroPosterior and Laya rows.

## 2. Against published models we downloaded and ran

Same test photos, same metrics. These models were trained by their authors on Nutrition5k
only, so DFood-TJ and NutritionVerse-Real measure transfer to a new kind of photo for them,
not a like-for-like comparison.

| model | Nutrition5k | never-seen | DFood-TJ | NV-Real |
|---|---|---|---|---|
| OmniFood8K method (CVPR 2026, official weights) | **36.8** | **50.5** | 168.9 | 639.3 |
| LayaMacro MacroPosterior (ours) | 41.4 | 56.7 | **57.0** | 130.3 |
| LayaMacro Laya graft (ours) | 45.7 | 59.8 | 61.0 | **121.6** |
| FoodLMM (2023, official weights, 7B) | 48.6 | 66.2 | 209.3 | 684.7 |
| Food-R1 (2026, official weights, 8B; kcal only) | 89.7 | 116.5 | 116.3 | 447.6 |

- On Nutrition5k, **OmniFood8K is best**; our best is second, 4.6 kcal behind. OmniFood8K
  fine-tunes two full backbones (287 M parameters) and adds a predicted depth map. Its
  training script keeps the best of 150 epochs by test-set score, which flatters its number
  by roughly 2-3 kcal.
- Two downloadable models are **not ranked** because they were trained on Nutrition5k test
  dishes: third-party DPF-Nutrition weights (18.4 kcal, the same as on its training dishes)
  and CalorieCLIP (its card describes a random split of all dishes).

## 3. Photos unlike any training data (no model trained on these)

Food Portion Benchmark (Central Asian lab plates) and LeFood-Set (Indonesian hospital
plates). **Grams error (weighed)**, lower is better:

| model | FPB | LeFood, before eating | LeFood, leftovers | 80 % coverage |
|---|---|---|---|---|
| Laya graft, 4 blocks trained | **194** | 75 | 69 | 0.09-0.24 |
| Laya graft, frozen | 216 | **61** | **55** | 0.14-0.25 |
| MacroPosterior, 12 blocks trained | 224 | 78 | 67 | 0.09-0.20 |
| OmniFood8K (CVPR 2026) | 283 | 78 | 61 | — |
| *no image* | *214* | *151* | *177* | — |

- **Our best model beats OmniFood8K in every column** (not every one of our models does:
  on LeFood leftovers OmniFood8K's 61 g beats MacroPosterior and Laya with trained blocks),
  and **Laya generalises best**.
- **Every model is poor at portion size on unfamiliar photos.** FPB plates (median 230 g) are
  predicted at ~95 g by every model (OmniFood8K: 39 g), and LeFood's single items (median
  46 g) at ~130 g: each model guesses the portion sizes of its own training data. A single
  photo carries no absolute scale.
- **Uncertainty ranges are over-confident here**: they contain the truth only 9-25 % of the
  time, against 80-84 % on the training datasets. Treat ranges on unfamiliar kinds of photos
  as too narrow.

## 4. Food questions (Laya)

Accuracy on test photos, pooled over the three datasets (the threshold column: Nutrition5k,
where always answering "no" scores 0.84). Yes/no food questions: always giving the most common
answer scores 0.55-0.63; "which of 5" has chance 0.20.

| model | contains X? (trained foods) | contains X? (never-trained foods) | which of 5? (trained) | which of 5? (never-trained) | over 500 kcal / 30 g protein…? (Nutrition5k) |
|---|---|---|---|---|---|
| Laya, multi-task, seed 0 | **0.89** | 0.65 | **0.86** | 0.40 | **0.92** |
| Laya, multi-task, seed 1 | 0.59 | 0.61 | 0.31 | 0.23 | 0.84 |
| Laya, nutrition only | 0.55 | 0.60 | 0.19 | 0.20 | 0.76 |
| Qwen3.5-2B, untrained | 0.80 | **0.83** | 0.62 | **0.55** | 0.61 |
| Gemma 4 E2B, untrained | 0.74 | 0.75 | 0.52 | 0.51 | 0.50 |

- **Laya can learn to recognise food** (seed 0: 0.89 / 0.86), and it uses the image: asked
  about a *different* meal's photo, accuracy drops to 0.64 / 0.41. A question phrasing never
  seen in training works as well as trained ones (0.88).
- **New foods transfer only partly.** For food names never trained on, an untrained VLM with
  broad pre-training is better (0.83 vs 0.65).
- **Training is unstable**: seed 1 of the same recipe failed to learn the new questions, and
  multi-task training made calorie estimates worse in both seeds (Nutrition5k 49.1 vs 46.0).
  The released Laya checkpoint is therefore the nutrition-only one.

## 5. What training adds

Untrained models, as downloaded, on Nutrition5k (kcal MAE): Gemma 4 E2B 310, Qwen3.5-2B 417,
Laya without an image 482, all worse than guessing without the photo (200). Training brings
every family to 41-57 kcal: a 6-10x error reduction.

## 6. Where the error comes from

Calories = grams × calories per gram. Replacing one predicted factor with the truth shows its
share of the error: perfect grams would cut calorie error by only 11-15 %, perfect calories per
gram by up to 60 % (DFood-TJ). **Recognising what is on the plate matters more than portion
size** on these datasets, which is why training the image encoder (48.1 → 41.4 kcal) helped
most.

## Negative results, kept on purpose

- Training more than 4 SigLIP 2 blocks inside Laya did not help: with 8 blocks, the two seeds
  gave 43.4 and 50.6 kcal.
- Training Laya for 20 instead of 10 epochs shifted which dataset it fits, not the average.
- Multi-task training cost calorie accuracy (above).
- On unfamiliar photos the uncertainty ranges are far too narrow (section 3).
