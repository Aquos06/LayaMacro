"""One evaluation protocol for every model, ours and the baselines alike.

Every model writes `runs/<name>/pred.npz` + `meta.json` through `save_predictions`: raw
logits over the frozen bins for each of the 5 targets, for every **calibration and test**
sample.
`evaluate_run` then treats all models identically:

1. one temperature per target, fitted on the calibration split (all datasets pooled). A
   model's own post-hoc calibration, if any, is ignored, so the comparison is fair;
2. every metric computed **per dataset test set**, never pooled, plus the Nutrition5k
   test_clean / test_novel subsets.

Point estimate = posterior mean of the calibrated distribution, as for MacroPosterior.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from layamacro.data.samples import DATASETS, Sample, load_samples
from layamacro.eval.metrics import (
    binary_metrics,
    classification_metrics,
    fit_temperature,
    interval_metrics,
    point_metrics,
)
from layamacro.posterior import distributions
from layamacro.schema import TARGETS, MacroBins, load_macro_bins

BINS = Path("configs/target_bins.yaml")
RUNS = Path("runs")
#: Never-trained yes/no questions answered from the distribution. Fixed before any run.
NOUL_V2 = {"kcal": 500.0, "protein_g": 30.0, "carbs_g": 40.0, "fat_g": 20.0, "mass_g": 400.0}
NOMINAL = 0.8


def load_bins() -> dict[str, MacroBins]:
    return load_macro_bins(BINS, required=TARGETS)


def git_commit() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
    ).stdout.strip()
    return head + ("-dirty" if dirty else "")


def eval_samples(samples: list[Sample] | None = None) -> list[Sample]:
    """The samples every run must predict: calibration + test of all datasets."""
    samples = samples if samples is not None else load_samples(check_images=False)
    return [s for s in samples if s.split in ("calibration", "test")]


def save_predictions(
    name: str, ids: list[str], logits: dict[str, np.ndarray], meta: dict[str, Any]
) -> Path:
    """logits[target] is [N, n_bins] raw (pre-temperature) logits, rows aligned with ids."""
    bins = load_bins()
    for t in TARGETS:
        if logits[t].shape != (len(ids), bins[t].n_bins):
            raise ValueError(f"{t}: logits {logits[t].shape} != ({len(ids)}, {bins[t].n_bins})")
        if not np.isfinite(logits[t]).all():
            raise ValueError(f"{t}: non-finite logits")
    need = {s.id for s in eval_samples()}
    missing = need - set(ids)
    if missing:
        raise ValueError(f"{len(missing)} calibration/test samples have no prediction")
    run_dir = RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"ids": np.array(ids)}
    arrays |= {f"logits_{t}": logits[t].astype(np.float32) for t in TARGETS}
    np.savez_compressed(run_dir / "pred.npz", **arrays)  # type: ignore[arg-type]
    meta = {"name": name, "commit": git_commit(), "bins": str(BINS), **meta}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    meta = json.loads((run_dir / "meta.json").read_text())
    pred = np.load(run_dir / "pred.npz")
    row = {str(i): k for k, i in enumerate(pred["ids"])}
    bins = load_bins()
    mbins = [bins[t] for t in TARGETS]
    samples = eval_samples()

    def logits_for(subset: list[Sample]) -> list[torch.Tensor]:
        idx = [row[s.id] for s in subset]
        return [torch.from_numpy(pred[f"logits_{t}"][idx]) for t in TARGETS]

    def true_bins(subset: list[Sample], t: str) -> torch.Tensor:
        return torch.tensor([bins[t].index_of(s.targets[t]) for s in subset])

    calib = [s for s in samples if s.split == "calibration"]
    cal_lg = logits_for(calib)
    temps = [fit_temperature(cal_lg[j], true_bins(calib, t)) for j, t in enumerate(TARGETS)]

    subsets: dict[str, list[Sample]] = {}
    for d in DATASETS:
        subsets[d] = [s for s in samples if s.dataset == d and s.split == "test"]
    for tag in ("test_clean", "test_novel"):
        subsets[f"nutrition5k_{tag.removeprefix('test_')}"] = [s for s in samples if tag in s.tags]

    results: dict[str, Any] = {}
    for name, subset in subsets.items():
        lg = logits_for(subset)
        raw = distributions(lg, mbins)
        cal = distributions(lg, mbins, temps)
        res: dict[str, Any] = {"n": len(subset)}
        for j, t in enumerate(TARGETS):
            y = np.array([s.targets[t] for s in subset])
            tb = true_bins(subset, t)
            lo, hi = cal[j].interval(NOMINAL)
            above = cal[j].prob_above(NOUL_V2[t]).numpy()
            res[t] = {
                "point": point_metrics(cal[j].expected().numpy(), y),
                "range_80": interval_metrics(lo.numpy(), hi.numpy(), y, NOMINAL),
                "bins": classification_metrics(cal[j].probs, tb),
                "bins_raw_ece": classification_metrics(raw[j].probs, tb)["ece"],
                f"over_{NOUL_V2[t]:g}": binary_metrics(above, (y > NOUL_V2[t]).astype(int)),
            }
        res["summary"] = summarize(res)
        results[name] = res
    return {
        "meta": meta,
        "temperatures": dict(zip(TARGETS, temps, strict=True)),
        "results": results,
    }


def summarize(res: dict[str, Any]) -> dict[str, float]:
    """Averages over the 5 targets — the one-line-per-model view."""

    def avg(*path: str) -> float:
        vals = []
        for t in TARGETS:
            x: Any = res[t]
            for p in path:
                x = x[p]
            vals.append(float(x))
        return float(np.mean(vals))

    def avg_noul(key: str) -> float:
        return float(np.mean([res[t][f"over_{NOUL_V2[t]:g}"][key] for t in TARGETS]))

    return {
        "kcal_mae": res["kcal"]["point"]["mae"],
        "kcal_r": res["kcal"]["point"]["pearson_r"],
        "kcal_medae_pct": res["kcal"]["point"]["medae_pct"],
        "mean_r": avg("point", "pearson_r"),
        "mean_medae_pct": avg("point", "medae_pct"),
        "accuracy": avg("bins", "accuracy"),
        "off_by_one": avg("bins", "off_by_one"),
        "macro_precision": avg("bins", "macro_precision"),
        "macro_recall": avg("bins", "macro_recall"),
        "macro_f1": avg("bins", "macro_f1"),
        "weighted_f1": avg("bins", "weighted_f1"),
        "macro_precision_pm1": avg("bins", "macro_precision_pm1"),
        "macro_recall_pm1": avg("bins", "macro_recall_pm1"),
        "macro_f1_pm1": avg("bins", "macro_f1_pm1"),
        "ece": avg("bins", "ece"),
        "coverage_80": avg("range_80", "coverage"),
        "yesno_precision": avg_noul("precision"),
        "yesno_recall": avg_noul("recall"),
        "yesno_f1": avg_noul("f1"),
        "yesno_auroc": avg_noul("auroc"),
    }
