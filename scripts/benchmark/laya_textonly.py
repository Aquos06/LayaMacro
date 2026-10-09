"""Untrained Laya (as downloaded, no image) on the 5 nutrition questions: a baseline.

Laya cannot see a photo without the trained graft, so its untrained answer is the same for
every photo: what the question text alone suggests. Written as an ordinary run
(runs/laya_pure_textonly) so scripts/benchmark/evaluate.py scores it like every model.

    uv run python scripts/benchmark/laya_textonly.py --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from question_bank import TextLayaProber

from layamacro.eval.evaluate import eval_samples, save_predictions
from layamacro.graft import INSTRUCTIONS, level_labels
from layamacro.schema import TARGETS, Question
from layamacro.train.common import load_data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    samples, bins = load_data()
    pr = TextLayaProber(torch.device(args.device))
    qs = [
        Question(id=t, kind="score", instructions=INSTRUCTIONS[t], options=level_labels(bins[t]))
        for t in TARGETS
    ]
    probs = pr.probs(qs, ["-"] * len(qs))
    ev = eval_samples(samples)
    logits = {t: np.tile(np.log(probs[j] + 1e-12), (len(ev), 1)) for j, t in enumerate(TARGETS)}
    run = save_predictions(
        "laya_pure_textonly",
        [s.id for s in ev],
        logits,
        {
            "method": "untrained Laya (laya-typed-decisions as downloaded): the 5 nutrition "
            "score questions with the text state 'A photo of a meal.' and no image; one answer "
            "for every photo",
            "trained": "no",
            "params_trainable": 0,
        },
    )
    for j, t in enumerate(TARGETS):
        print(f"{t}: most likely bin {int(np.argmax(probs[j]))} p={float(np.max(probs[j])):.2f}")
    print(f"wrote {run}")


if __name__ == "__main__":
    main()
