"""Score every model on the unseen-dataset benchmark: Food Portion Benchmark + LeFood-Set.

No model scored here trained on these datasets. Ours: runs with a `bench.npz` (the trainers'
`--bench`), temperatures fitted on the calibration split exactly as
layamacro.eval.evaluate does, point = expected value, range = central 80 %. Downloaded
models: runs/external/<name>_bench/points.npz (one point estimate per target).

Table 1 is **grams (weighed)**: the primary metric. Table 2 is **derived** kcal / macros
(weighed grams x composition table), reported separately. LeFood photos relying on a
low-confidence mapping row are also shown excluded.

    uv run python scripts/benchmark/unseen.py
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from layamacro.data.samples import Sample
from layamacro.eval.bench import bench_samples
from layamacro.eval.evaluate import eval_samples, load_bins
from layamacro.eval.metrics import fit_temperature
from layamacro.posterior import distributions
from layamacro.schema import TARGETS
from layamacro.train.common import load_data

OUT = Path("docs/results/unseen.md")
RUNS = Path("runs")


def discover() -> tuple[dict[str, str], dict[str, str]]:
    """Our runs with a bench.npz (grouped over seeds) and downloaded models' benchmark points
    (runs/external/<name>/points.npz with a `_bench` suffix)."""
    ours = {
        re.sub(
            r"_s\d+$", "", p.parent.name
        ): f"**ours** · {re.sub(r'_s[0-9]+$', '', p.parent.name)}"
        for p in sorted(RUNS.glob("*/bench.npz"))
    }
    external = {
        p.parent.name: f"external · {p.parent.name.removesuffix('_bench')}"
        for p in sorted((RUNS / "external").glob("*_bench/points.npz"))
    }
    return ours, external


SUBSETS: dict[str, Callable[[Sample], bool]] = {
    "FPB (Central Asian, lab)": lambda s: s.dataset == "food_portion_benchmark",
    "LeFood before eating": lambda s: s.dataset == "lefood_set" and "before" in s.tags,
    "LeFood after eating (leftovers)": lambda s: s.dataset == "lefood_set" and "after" in s.tags,
}
MACROS = ("kcal", "protein_g", "carbs_g", "fat_g")


def ours(group: str, bs: list[Sample]) -> list[dict[str, np.ndarray]]:
    """Per seed: {target: point, target_lo, target_hi} aligned with bs."""
    bins = load_bins()
    mb = [bins[t] for t in TARGETS]
    calib = [s for s in eval_samples() if s.split == "calibration"]
    out = []
    for run in sorted(RUNS.glob(f"{group}_s*")):
        if not re.fullmatch(rf"{re.escape(group)}_s\d+", run.name):
            continue
        pred, bench = np.load(run / "pred.npz"), np.load(run / "bench.npz")
        prow = {str(i): k for k, i in enumerate(pred["ids"])}
        brow = {str(i): k for k, i in enumerate(bench["ids"])}
        ci = [prow[s.id] for s in calib]
        bi = [brow[s.id] for s in bs]
        temps = [
            fit_temperature(
                torch.from_numpy(pred[f"logits_{t}"][ci]),
                torch.tensor([bins[t].index_of(s.targets[t]) for s in calib]),
            )
            for t in TARGETS
        ]
        dist = distributions(
            [torch.from_numpy(bench[f"logits_{t}"][bi]) for t in TARGETS], mb, temps
        )
        r: dict[str, np.ndarray] = {}
        for j, t in enumerate(TARGETS):
            r[t] = dist[j].expected().numpy()
            lo, hi = dist[j].interval(0.8)
            r[f"{t}_lo"], r[f"{t}_hi"] = lo.numpy(), hi.numpy()
        out.append(r)
    return out


def external(name: str, bs: list[Sample]) -> list[dict[str, np.ndarray]]:
    d = np.load(RUNS / "external" / name / "points.npz")
    row = {str(i): k for k, i in enumerate(d["ids"])}
    idx = [row[s.id] for s in bs]
    return [{t: d[t][idx].astype(np.float64) for t in TARGETS if t in d.files}]


def scores(
    seeds: list[dict[str, np.ndarray]], bs: list[Sample], t: str, sel: np.ndarray
) -> dict[str, float]:
    """Mean over seeds of MAE, MAE/mean %, r and 80 % coverage on rows `sel` with target t."""
    y = np.array([s.targets.get(t, np.nan) for s in bs])
    m = sel & ~np.isnan(y)
    res: dict[str, list[float]] = {"mae": [], "pct": [], "r": [], "cov": []}
    for p in seeds:
        e = np.abs(p[t][m] - y[m])
        res["mae"].append(float(e.mean()))
        res["pct"].append(100 * float(e.mean() / y[m].mean()))
        if np.ptp(p[t][m]) > 1e-6:  # a constant guess has no correlation
            res["r"].append(float(np.corrcoef(p[t][m], y[m])[0, 1]))
        if f"{t}_lo" in p:
            res["cov"].append(float(((p[f"{t}_lo"][m] <= y[m]) & (y[m] <= p[f"{t}_hi"][m])).mean()))
    out = {k: float(np.mean(v)) for k, v in res.items() if v}
    out["n"] = int(m.sum())
    return out


def bold_table(
    head: list[str], rows: list[tuple[str, list[float | None]]], picks: list[str], fmts: list[str]
) -> list[str]:
    best: list[float | None] = []
    for j, pick in enumerate(picks):
        vals: list[float] = [
            v for lbl, r in rows if (v := r[j]) is not None and not lbl.startswith("Baseline")
        ]
        if not vals or pick == "":
            best.append(None)
        elif pick == "near80":
            best.append(min(vals, key=lambda v: abs(v - 0.8)))
        else:
            best.append(min(vals) if pick == "min" else max(vals))
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, row_vals in rows:
        cells = []
        for j, v in enumerate(row_vals):
            if v is None:
                cells.append("—")
            else:
                s = fmts[j].format(v)
                cells.append(f"**{s}**" if best[j] is not None and v == best[j] else s)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    bs = bench_samples()
    samples, _ = load_data()
    train = [s for s in samples if s.split == "train"]
    prior = {t: float(np.median([s.targets[t] for s in train])) for t in TARGETS}
    models: list[tuple[str, list[dict[str, np.ndarray]], str]] = []
    OURS, EXTERNAL = discover()
    for g, label in OURS.items():
        runs = ours(g, bs)
        if runs:
            models.append((label, runs, f"{len(runs)} seed{'s' * (len(runs) > 1)}"))
    for name, label in EXTERNAL.items():
        if (RUNS / "external" / name / "points.npz").exists():
            models.append((f"external · {label}", external(name, bs), "downloaded"))
    models.append(
        (
            "Baseline: training median (no image)",
            [{t: np.full(len(bs), prior[t]) for t in TARGETS}],
            "no image",
        )
    )
    sel = {k: np.array([f(s) for s in bs]) for k, f in SUBSETS.items()}
    lowconf = np.array(["lowconf" in s.tags for s in bs])
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    L = [
        "# Unseen-dataset benchmark: datasets no model trained on",
        "",
        f"Generated by `scripts/benchmark/unseen.py` at commit `{commit}`. "
        "Every row meets these photos for the first time: our default models never trained on "
        "the Food Portion Benchmark or LeFood-Set, and neither did OmniFood8K. This is the "
        "fair head-to-head that DFood-TJ and NutritionVerse-Real could not be.",
        "",
        "- **FPB:** Central Asian plates in a lab, 11,101 photos of the authors' train + val "
        "splits; each plate is shown from several angles, so photos are not independent.",
        "- **LeFood-Set:** Indonesian hospital plates, one food item per photo, before eating "
        "(524) and the weighed leftovers after eating (317, partly eaten).",
        "- MAE % = MAE ÷ mean(true). **Bold = best**; the no-image baseline is excluded from "
        "bolding. Coverage: share of true values inside the 80 % range (should be ≈ 0.80; ours "
        "only).",
        "",
        "## 1. Grams (weighed): the primary benchmark",
        "",
    ]
    head = ["model", "runs"]
    picks: list[str] = [""]
    fmts = ["{}"]
    for k in SUBSETS:
        head += [f"{k} MAE g ↓", "MAE % ↓", "r ↑", "cov 80 %"]
        picks += ["min", "min", "max", "near80"]
        fmts += ["{:.1f}", "{:.1f}", "{:.3f}", "{:.2f}"]
    rows = []
    for label, seeds, how in models:
        vals: list[Any] = [how]
        for k in SUBSETS:
            sc = scores(seeds, bs, "mass_g", sel[k])
            vals += [sc["mae"], sc["pct"], sc.get("r"), sc.get("cov")]
        rows.append((label, vals))
    ns = {k: int(sel[k].sum()) for k in SUBSETS}
    L += bold_table(
        head,
        [(lbl, [None if isinstance(v, str) else v for v in vals]) for lbl, vals in rows],
        picks,
        fmts,
    )
    # put the run counts back (bold_table treats them as missing)
    for i, (_, _, how) in enumerate(models):
        L[-len(models) + i] = L[-len(models) + i].replace("| — |", f"| {how} |", 1)
    L += ["", "Photos: " + ", ".join(f"{k} n = {n}" for k, n in ns.items()) + "."]

    L += [
        "",
        "## 2. Calories and macros (derived, not weighed)",
        "",
        "Weighed grams x a composition-table row per food (`configs/nutrition_mapping/`). These "
        "inherit the table's error, so they rank models against the same imperfect reference "
        "rather than measure absolute accuracy. FPB: only plates whose every item is mapped. "
        "LeFood: all 841 photos; the last two columns drop the 178 relying on a low-confidence "
        "row (e.g. bone-in chicken weighed with the bone).",
        "",
    ]
    fpb = sel["FPB (Central Asian, lab)"]
    lef = sel["LeFood before eating"] | sel["LeFood after eating (leftovers)"]
    groups = {"FPB": fpb, "LeFood": lef, "LeFood, high/medium only": lef & ~lowconf}
    head = ["model", "runs"]
    picks, fmts = [""], ["{}"]
    for gname in groups:
        head += [
            f"{gname} kcal MAE ↓",
            "kcal r ↑",
            "protein MAE g ↓",
            "carbs MAE g ↓",
            "fat MAE g ↓",
        ]
        picks += ["min", "max", "min", "min", "min"]
        fmts += ["{:.1f}", "{:.3f}", "{:.1f}", "{:.1f}", "{:.1f}"]
    rows2: list[tuple[str, list[float | None]]] = []
    for label, seeds, _how in models:
        vals2: list[float | None] = [None]
        for gsel in groups.values():
            kc = scores(seeds, bs, "kcal", gsel)
            vals2 += [kc["mae"], kc.get("r")]
            vals2 += [scores(seeds, bs, t, gsel)["mae"] for t in MACROS[1:]]
        rows2.append((label, vals2))
    L += bold_table(head, rows2, picks, fmts)
    for i, (_, _, how) in enumerate(models):
        L[-len(models) + i] = L[-len(models) + i].replace("| — |", f"| {how} |", 1)
    n_fpb = int((fpb & np.array(["kcal" in s.targets for s in bs])).sum())
    L += [
        "",
        f"Photos with derived values: FPB n = {n_fpb}, LeFood n = {int(lef.sum())} "
        f"({int((lef & ~lowconf).sum())} high/medium).",
    ]
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
