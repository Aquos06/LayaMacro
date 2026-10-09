"""Estimate calories and macros for meal photos with a trained LayaMacro model.

    uv run python scripts/infer/predict.py --weights weights/macroposterior.pt meal.jpg
    uv run python scripts/infer/predict.py --weights weights/laya.pt meal.jpg \\
        --ask "Does this meal contain rice?" --options yes no

Prints each nutrient as `value (80 % range: low-high)`. `--json` prints machine-readable
output instead. `--ask` (Laya models only) asks any question with 2-8 options, without
retraining; its probabilities are raw (not calibrated).

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
    ap.add_argument("photos", nargs="+", type=Path)
    ap.add_argument("--weights", type=Path, required=True, help="a LayaMacro model.pt")
    ap.add_argument("--device", default=None, help="cuda, cuda:1, cpu (default: cuda if available)")
    ap.add_argument("--ask", default=None, help="Laya only: an extra question about each photo")
    ap.add_argument("--options", nargs="+", default=["yes", "no"], help="answers for --ask")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    model = load_predictor(args.weights, args.device)
    results = model.predict(list(args.photos))
    answers = None
    if args.ask:
        if not isinstance(model, LayaPredictor):
            ap.error("--ask needs a Laya model (MacroPosterior only predicts nutrition)")
        answers = model.ask(list(args.photos), args.ask, args.options)

    if args.json:
        out = [
            {"photo": str(p), "nutrition": r} | ({"answer": answers[i]} if answers else {})
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
        if answers:
            best = max(answers[i], key=answers[i].__getitem__)
            probs = ", ".join(f"{o} {pr:.0%}" for o, pr in answers[i].items())
            print(f"  {args.ask} -> {best} ({probs})")


if __name__ == "__main__":
    main()
