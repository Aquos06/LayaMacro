# LayaMacro

**One photo of a meal → calories, protein, carbohydrate, fat and weight, each with an honest
uncertainty range.** LayaMacro grafts a vision encoder onto
[Laya](https://huggingface.co/convaiinnovations/laya), an encoder-only *typed decision*
model, so it answers nutrition questions as calibrated probability distributions instead of
generated text.

> Status: being assembled from the research code. Not yet released.

## Layout (planned)

```
src/layamacro/      model code: vision encoder, projector, Laya graft, MacroPosterior head
scripts/train/      training: MacroPosterior, Laya graft (nutrition-only and multi-task)
scripts/infer/      run a trained model on your own photo
scripts/benchmark/  evaluation: test sets, unseen-dataset benchmark, food-question bank,
                    comparison with untrained VLMs and published models
configs/            frozen bin edges and question bank settings
docs/               architecture, results, datasets
tests/              unit tests (no data needed) and data tests
```

## Licence

- **Code:** Apache License 2.0 (`LICENSE`).
- **Trained weights:** CC BY-NC-SA 4.0, non-commercial (`MODEL_LICENSE.md` explains why).
- **Credits** for the models and datasets this builds on: `NOTICE`.

Built on Laya by convaiinnovations and SigLIP 2 by Google. **Not affiliated with or endorsed
by** either. Not a medical device.
