# Contributing to LayaMacro

Thanks for your interest. LayaMacro is a research project: a small codebase, a released
checkpoint and benchmarks whose numbers we try to report honestly, negative ones included.
Contributions that keep it that way are very welcome.

## What helps most

- **Bug reports and fixes** in loading, inference, training or the benchmark scripts.
- **Reproductions.** Training is unstable across seeds (README, Limitations). A run of the
  released recipe with another seed, good or bad, is useful: open an issue with the numbers.
- **Prediction reports** on meals and photos unlike the training data: other cuisines, home
  cooking, phone photos at an angle. Use the *Poor prediction* issue template.
- **New evaluation datasets or baselines**, under a licence that allows it (below).
- **Documentation** that is wrong, unclear or out of date.

For anything bigger than a small fix (a change to the model, the training recipe or a
reported number), please open an issue first so we can agree on the approach before you
spend GPU hours on it.

## Set up

Python 3.12 and [uv](https://docs.astral.sh/uv/). Fork the repository, then:

```bash
git clone https://github.com/<your-username>/LayaMacro && cd LayaMacro
uv sync --extra train --extra dev
```

## Checks

These must pass before a pull request is merged:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -m "not slow and not data"
```

The fast tests need no GPU, weights or data. Tests marked `slow` need the weights or a GPU,
and tests marked `data` need the downloaded datasets ([docs/datasets.md](docs/datasets.md)).
Run the full `uv run pytest` if you change model loading, inference or a dataset loader.

## Code

- Match the surrounding code: its naming, comment density and idioms. The `ruff` and `mypy`
  settings are in `pyproject.toml` (line length 100, Python 3.12, type hints throughout).
- Keep pull requests focused: one change per PR is easier to review and to revert.
- Commit messages: a short imperative summary (`Load downloaded checkpoints with
  torch.load(weights_only=True)`), with a body that says why when it is not obvious.
- Add or update a test when you fix a bug or change behaviour.

## Results and claims

The numbers in the README and [docs/results.md](docs/results.md) come from the benchmark
scripts. If your change moves them:

- give the exact commands, the seed(s) and the checkpoint you ran;
- report every test set the benchmark covers, not only the ones that improved;
- run more than one seed where you can (the recipe is seed-sensitive), and say so when you
  could not;
- update `docs/results.md`, and the README tables if the released checkpoint changes.

A careful negative result is as welcome as an improvement.

## Data, images and licences

- **Never commit** images, datasets, cached features or weights. `.gitignore` blocks the
  usual extensions; please do not force-add around it.
- Meal photos can be personal data. Only attach a photo to an issue or pull request if you
  took it yourself, are happy for it to be public, and it shows no people or identifying
  details. **Never** share images from DFood-TJ-F-Nutr: they are photos of hospital
  patients' meals.
- A new dataset or model needs a licence that permits how we use it. Add it to `NOTICE` and
  `docs/datasets.md`. If it is non-commercial or share-alike and used for training, it also
  affects the weights licence ([MODEL_LICENSE.md](MODEL_LICENSE.md)): say so in the PR.

## Licence of contributions

By contributing you agree that your contribution is licensed under the Apache License 2.0,
like the rest of the source code ([LICENSE](LICENSE), section 5). Weights trained with this
code and released by the project are licensed separately ([MODEL_LICENSE.md](MODEL_LICENSE.md)).

## Conduct and security

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
Report security vulnerabilities privately as described in [SECURITY.md](SECURITY.md), not
in a public issue.
