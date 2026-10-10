"""Estimate calories and macros for meal photos with LayaMacro, and ask about the food.

    uv run python scripts/infer/predict.py --weights hf://Aquos06/LayaMacro/layamacro-laya.pt meal.jpg
    ... meal.jpg --ask "Does this meal contain rice?" --options yes no
    ... meal.jpg --over kcal=500 protein_g=30
    ... --foods                      # the food names the model was trained to recognise

Each nutrient is printed as `value (80 % range: low-high)`; `--json` prints machine-readable
output. `--over` gives calibrated probabilities from the nutrition distribution. `--ask`
answers any question with 2-8 options; it is reliable for the question kinds and foods the
model was trained on (see `--foods` and docs/results.md), weak otherwise, and its
probabilities are not calibrated.

Not a medical device: see MODEL_LICENSE.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from layamacro.inference import LayaPredictor, load_predictor

LABELS = {
    "kcal": ("energy", "kcal"),
    "protein_g": ("protein", "g"),
    "carbs_g": ("carbohydrate", "g"),
    "fat_g": ("fat", "g"),
    "mass_g": ("food weight", "g"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("photos", nargs="*", type=Path)
    ap.add_argument(
        "--weights",
        required=True,
        help="a LayaMacro checkpoint: a local file or hf://<user>/<repo>/<file>",
    )
    ap.add_argument("--device", default=None, help="cuda, cuda:1, cpu (default: cuda if available)")
    ap.add_argument("--ask", default=None, help="Laya only: an extra question about each photo")
    ap.add_argument("--options", nargs="+", default=["yes", "no"], help="answers for --ask")
    ap.add_argument(
        "--over", nargs="+", default=[], metavar="TARGET=VALUE", help="e.g. kcal=500 protein_g=30"
    )
    ap.add_argument(
        "--foods", action="store_true", help="list the foods the model knows, then exit"
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    model = load_predictor(args.weights, args.device)
    if args.foods:
        foods = getattr(model, "foods", [])
        print("\n".join(foods) if foods else "this checkpoint was not trained on food questions")
        return
    if not args.photos:
        ap.error("give at least one photo")
    results = model.predict(list(args.photos))
    overs: dict[str, list[float]] = {}
    for spec in args.over:
        target, value = spec.split("=")
        if target not in LABELS:
            ap.error(f"--over: unknown target {target!r}; use one of {', '.join(LABELS)}")
        if not isinstance(model, LayaPredictor):
            ap.error("--over needs a Laya model")
        overs[spec] = model.prob_over(list(args.photos), target, float(value))
    answers = None
    if args.ask:
        if not isinstance(model, LayaPredictor):
            ap.error("--ask needs a Laya model (MacroPosterior only predicts nutrition)")
        answers = model.ask(list(args.photos), args.ask, args.options)

    if args.json:
        out = [
            {"photo": str(p), "nutrition": r}
            | ({"answer": answers[i]} if answers else {})
            | ({"over": {k: v[i] for k, v in overs.items()}} if overs else {})
            for i, (p, r) in enumerate(zip(args.photos, results, strict=True))
        ]
        print(json.dumps(out, indent=2))
        return
    for i, (p, r) in enumerate(zip(args.photos, results, strict=True)):
        print(f"\n{p}")
        for t, (name, unit) in LABELS.items():
            v = r[t]
            print(
                f"  {name:13s} {v['value']:7.1f} {unit:4s} (80 % range: {v['low']:.0f}-{v['high']:.0f})"
            )
        for spec, pr in overs.items():
            target, value = spec.split("=")
            name, unit = LABELS[target]
            print(f"  P({name} > {value} {unit}) = {pr[i]:.0%}")
        if answers:
            best = max(answers[i], key=answers[i].__getitem__)
            probs = ", ".join(f"{o} {pr:.0%}" for o, pr in answers[i].items())
            print(f"  {args.ask} -> {best} ({probs})")


if __name__ == "__main__":
    main()
