"""Metric definitions. Every reported number is computed here; don't improvise elsewhere.

MAE is primary. Both MAE% definitions are reported, labelled, with `n` and `mean(true)`:
`mae_pct_sample` is the mean of per-sample relative errors, `mae_pct_n5k` is
MAE / mean(true), Nutrition5k's own definition. Correlation is always reported, because the mean predictor has a
respectable MAE and a correlation of zero. Coverage is never reported without width.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


def point_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = np.abs(pred - true)
    rel = err[true > 0] / true[true > 0]
    r = float(np.corrcoef(pred, true)[0, 1]) if np.ptp(pred) > 0 else 0.0
    return {
        "n": float(len(true)),
        "mean_true": float(true.mean()),
        "mae": float(err.mean()),
        "mae_pct_sample": float(100 * rel.mean()),
        "mae_pct_n5k": float(100 * err.mean() / true.mean()),
        "medae_pct": float(100 * np.median(rel)),
        "pearson_r": r,
    }


def interval_metrics(
    lo: np.ndarray, hi: np.ndarray, true: np.ndarray, nominal: float
) -> dict[str, float]:
    """Coverage *and* width together; plus the interval score (lower is better)."""
    alpha = 1 - nominal
    width = hi - lo
    score = (
        width + (2 / alpha) * (lo - true) * (true < lo) + (2 / alpha) * (true - hi) * (true > hi)
    )
    pos = true > 0
    return {
        "nominal": nominal,
        "coverage": float(((lo <= true) & (true <= hi)).mean()),
        "mean_width": float(width.mean()),
        "median_width_pct": float(100 * np.median(width[pos] / true[pos])),
        "interval_score": float(score.mean()),
    }


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with **equal-mass** (equal-count) bins."""
    order = np.argsort(confidence)
    total = len(confidence)
    out = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            out += len(chunk) / total * abs(correct[chunk].mean() - confidence[chunk].mean())
    return float(out)


def bin_metrics(probs: Tensor, true_bin: Tensor) -> dict[str, float]:
    """score-question metrics: exact, off-by-one, bin MAE, NLL, ECE on answer_confidence."""
    top = probs.argmax(-1)
    conf = probs.max(-1).values
    diff = (top - true_bin).abs()
    nll = -torch.log(probs.gather(1, true_bin[:, None]).clamp_min(1e-12)).mean()
    return {
        "exact": float((diff == 0).float().mean()),
        "off_by_one": float((diff <= 1).float().mean()),
        "bin_mae": float(diff.float().mean()),
        "nll": float(nll),
        "ece": ece(conf.numpy(), (diff == 0).numpy().astype(float)),
    }


def fit_temperature(logits: Tensor, target_bin: Tensor) -> float:
    """Upstream Laya's recipe (docs/laya.md): LBFGS on log T, clamped to [0.1, 10].

    Call it on the calibration split only. Fitting on test and reporting ECE on test is the
    classic way calibration results get faked.
    """
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    lg = logits.float()

    def closure() -> Tensor:
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), target_bin)
        loss.backward()
        return loss

    opt.step(closure)
    return float(min(max(math.exp(log_t.item()), 0.1), 10.0))


def classification_metrics(probs: Tensor, true_bin: Tensor) -> dict[str, float]:
    """The bin prediction as a K-way classification: accuracy, off-by-one, and macro
    precision / recall / F1 over the bins present in the truth, plus weighted F1."""
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    base = bin_metrics(probs, true_bin)
    y, p = true_bin.numpy(), probs.argmax(-1).numpy()
    labels = sorted(set(y.tolist()))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, p, labels=labels, average="macro", zero_division=0
    )
    return {
        "accuracy": base["exact"],
        "off_by_one": base["off_by_one"],
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "weighted_f1": float(f1_score(y, p, labels=labels, average="weighted", zero_division=0)),
        **tolerant_prf(y, p, labels, tolerance=1),
        "bin_mae": base["bin_mae"],
        "nll": base["nll"],
        "ece": base["ece"],
    }


def tolerant_prf(
    y: np.ndarray, p: np.ndarray, labels: list[int], tolerance: int = 1
) -> dict[str, float]:
    """Macro precision / recall / F1 where a prediction within `tolerance` bins counts.

    Same macro averaging as the exact version (per bin, over the bins present in the
    truth), but "correct" means |predicted bin - true bin| <= tolerance:
    - precision_k: of photos predicted in bin k, the share whose truth is within ±1 of k;
    - recall_k: of photos truly in bin k, the share predicted within ±1 of k.
    Bins are x1.35-1.41 wide, so ±1 bin is "within roughly a factor of 1.4".
    """
    close = np.abs(p - y) <= tolerance
    precs, recs, f1s = [], [], []
    for k in labels:
        pred_k, true_k = p == k, y == k
        prec = float(close[pred_k].mean()) if pred_k.any() else 0.0
        rec = float(close[true_k].mean())
        precs.append(prec)
        recs.append(rec)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {
        "macro_precision_pm1": float(np.mean(precs)),
        "macro_recall_pm1": float(np.mean(recs)),
        "macro_f1_pm1": float(np.mean(f1s)),
    }


def binary_metrics(p_true: np.ndarray, label: np.ndarray) -> dict[str, float]:
    """A yes/no question thresholded at p = 0.5, plus threshold-free AUROC and Brier."""
    from sklearn.metrics import roc_auc_score

    pred = p_true >= 0.5
    tp = float((pred & (label == 1)).sum())
    fp = float((pred & (label == 0)).sum())
    fn = float((~pred & (label == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both = 0 < label.sum() < len(label)
    return {
        "base_rate": float(label.mean()),
        "accuracy": float((pred == (label == 1)).mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": float(roc_auc_score(label, p_true)) if both else math.nan,
        "brier": float(((p_true - label) ** 2).mean()),
    }
