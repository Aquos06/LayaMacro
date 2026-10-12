## What this changes

<!-- One or two sentences. Link the issue it addresses, e.g. "Fixes #12". -->

## Why

<!-- The problem it solves, or the result it improves. -->

## How it was checked

- [ ] `uv run ruff check .` and `uv run ruff format --check .`
- [ ] `uv run mypy`
- [ ] `uv run pytest -m "not slow and not data"`
- [ ] The full `uv run pytest`, if this touches model loading, inference or a dataset loader

## Results (only if reported numbers change)

<!--
The exact commands, seed(s) and checkpoint, and a before/after table for every test set
the benchmark covers, not only the ones that improved. Say if you ran a single seed.
-->

## Checklist

- [ ] No images, datasets, cached features or weights in the diff
- [ ] README / `docs/` updated where behaviour or numbers changed
- [ ] Any new dataset or model: licence checked, added to `NOTICE` and `docs/datasets.md`
- [ ] I have read [CONTRIBUTING.md](https://github.com/Aquos06/LayaMacro/blob/main/CONTRIBUTING.md) and agree to license this contribution under Apache-2.0
