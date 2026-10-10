"""Turn a training run into a release checkpoint: trained weights + calibration temperatures.

    uv run python scripts/export_weights.py --run runs/laya_ft4_s0 --out weights/layamacro-laya.pt

- **Temperatures**: one per target, fitted on the calibration split from the run's saved
  predictions (exactly as layamacro.eval.evaluate does), stored in the checkpoint so
  `layamacro.inference` gives calibrated ranges.
- **Only what we trained**: for a Laya graft, tensors identical to the downloaded Laya
  checkpoint are dropped; the loader takes them from the official Laya release. The export
  verifies that the slim checkpoint rebuilds a model whose state equals the full one.
- **Self-contained**: the bin edges travel with the weights, and training arguments are
  reduced to what the loader needs (no machine paths).

Released weights are CC BY-NC-SA 4.0 (MODEL_LICENSE.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from layamacro.eval.evaluate import evaluate_run, load_bins
from layamacro.schema import TARGETS

#: Training arguments the inference loader needs; everything else is dropped.
KEEP_ARGS = ("tune_last", "no_resampler", "n_image_tokens", "checkpoint", "scope", "multitask")


def slim_laya(state: dict[str, torch.Tensor], args: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Drop every tensor that equals the pristine Laya checkpoint's."""
    import laya

    from layamacro.graft import build_graft

    agent = laya.load(
        args.get("checkpoint", "convaiinnovations/laya-typed-decisions"), device="cpu"
    )
    fresh = build_graft(agent, 1024, 576, no_resampler=args.get("no_resampler", True)).float()
    base = fresh.state_dict()
    keep = {
        k: v
        for k, v in state.items()
        if not k.startswith("laya.") or k not in base or not torch.equal(v.float(), base[k].float())
    }
    # check: pristine + kept == the trained state, tensor for tensor
    fresh.load_state_dict(keep, strict=False)
    rebuilt = fresh.state_dict()
    bad = [k for k in state if not torch.equal(rebuilt[k].float(), state[k].float())]
    if bad:
        raise AssertionError(f"slim checkpoint does not rebuild the model: {bad[:5]}")
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="a run folder with model.pt + pred.npz")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    run_args = ckpt["args"] if isinstance(ckpt["args"], dict) else vars(ckpt["args"])
    ev = evaluate_run(args.run)
    temps = [float(ev["temperatures"][t]) for t in TARGETS]
    is_laya = "no_resampler" in run_args or "scope" in run_args
    out: dict[str, Any] = {
        "format": "layamacro-1",
        "kind": "laya" if is_laya else "macroposterior",
        "args": {k: run_args[k] for k in KEEP_ARGS if k in run_args},
        "temperatures": temps,
        "bins": {t: b.model_dump() for t, b in load_bins().items()},
        "test_summary": {
            k: {m: float(v) for m, v in r["summary"].items()} for k, r in ev["results"].items()
        },
    }
    if is_laya and run_args.get("multitask"):
        # The food names the question bank trained on (answered well) and those held out
        # (never trained on: answered weakly). Shipped so users know what to ask about.
        from layamacro.data.questions import vocab

        v = vocab()
        out["foods"] = sorted(set().union(*(known for known, _ in v.values())))
        out["foods_untrained"] = sorted(set().union(*(held for _, held in v.values())))
    if is_laya:
        out["model"] = slim_laya(ckpt["model"], run_args)
        out["tail"] = ckpt.get("tail")
        out["slim"] = True
    else:
        out["model"] = ckpt["model"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    n = sum(v.numel() for v in out["model"].values()) + sum(
        v.numel() for v in (out.get("tail") or {}).values()
    )
    print(f"wrote {args.out}: {n / 1e6:.1f}M parameters, {args.out.stat().st_size / 2**20:.0f} MiB")
    print("temperatures", dict(zip(TARGETS, [round(t, 3) for t in temps], strict=True)))


if __name__ == "__main__":
    main()
