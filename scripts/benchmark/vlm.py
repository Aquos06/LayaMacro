"""Benchmark untrained ("pure") vision-language models on the nutrition test sets.

Each nutrition target is one prompt listing the 18 bin ranges as options A-R; the model's
next-token probabilities over those 18 letters are its answer, a full distribution (no
generated text to parse). Writes runs/vlm_<model>_zeroshot/, scored like every other run by
scripts/benchmark/evaluate.py, and usable as a baseline by scripts/benchmark/question_bank.py.

    uv run python scripts/benchmark/vlm.py --model gemma4-e2b
    uv run python scripts/benchmark/vlm.py --model qwen3.5-2b
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from layamacro.baselines.vlm import MODELS, letter_ids, load_vlm, zero_shot_prompt
from layamacro.data.images import load_image
from layamacro.eval.evaluate import eval_samples, save_predictions
from layamacro.schema import TARGETS
from layamacro.train.common import load_data


def zeroshot(args: argparse.Namespace) -> None:
    samples, bins = load_data()
    vlm = load_vlm(args.model, args.device)
    ev = eval_samples(samples)
    prompts = {t: vlm.prompt(zero_shot_prompt(t, bins[t])) for t in TARGETS}
    letters = {t: torch.tensor(letter_ids(vlm, bins[t].n_bins)) for t in TARGETS}
    print(f"{len(ev)} images x {len(TARGETS)} questions; prompt example:\n{prompts['kcal'][-400:]}")

    loader: DataLoader[Any] = DataLoader(
        [s.image for s in ev],  # type: ignore[arg-type]
        batch_size=args.batch,
        num_workers=4,
        collate_fn=lambda paths: [load_image(p) for p in paths],
    )
    out: dict[str, list[np.ndarray]] = {t: [] for t in TARGETS}
    t0 = time.time()
    with torch.inference_mode():
        for k, images in enumerate(loader):
            for t in TARGETS:
                b = vlm.batch([prompts[t]] * len(images), images).to(args.device)
                # left-padded, so -1 is the answer slot; keep only it (a 262k vocab x every
                # position is ~2 GB per batch on Gemma)
                logits = vlm.model(**b, logits_to_keep=1).logits[:, -1].float()
                out[t].append(logits[:, letters[t].to(args.device)].cpu().numpy())
            if k % 20 == 0:
                done = (k + 1) * args.batch
                el = time.time() - t0
                print(
                    f"  {done}/{len(ev)}  ETA {(len(ev) - done) * el / done / 60:.0f} min",
                    flush=True,
                )
    ms = (time.time() - t0) / len(ev) * 1e3
    run = save_predictions(
        f"vlm_{args.model}_zeroshot",
        [s.id for s in ev],
        {t: np.concatenate(out[t]) for t in TARGETS},
        {
            "method": f"{MODELS[args.model]}, zero-shot (no training): one prompt per "
            "target listing the 18 ranges as A-R; next-token logits of the 18 letters = bin logits",
            "trained": "no (zero-shot)",
            "params_trainable": 0,
            "ms_per_image_5_questions": round(ms, 1),
        },
    )
    print(f"wrote {run}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    zeroshot(ap.parse_args())


if __name__ == "__main__":
    main()
