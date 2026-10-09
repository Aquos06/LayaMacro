"""Shared plumbing for the trainers: samples, targets, soft bin labels.

Every trainer builds its targets here, so "the same data" means the same arrays.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from layamacro.data.samples import Sample, load_samples
from layamacro.eval.evaluate import load_bins
from layamacro.posterior import hl_gauss_target
from layamacro.schema import TARGETS, MacroBins

#: Where trained weights are written. Override with LAYAMACRO_CHECKPOINTS (e.g. a larger disk);
#: the run folder always gets a `model.pt` link to the file.
CKPT_ROOT = Path(os.environ.get("LAYAMACRO_CHECKPOINTS", "checkpoints"))


def save_weights(run_dir: str | Path, obj: dict[str, Any]) -> Path:
    """torch.save `obj` to CKPT_ROOT/<run>/model.pt and link it as <run_dir>/model.pt."""
    run_dir = Path(run_dir)
    dest = CKPT_ROOT / run_dir.name / "model.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, dest)
    link = run_dir / "model.pt"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(dest.resolve())
    return dest


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data() -> tuple[list[Sample], dict[str, MacroBins]]:
    return load_samples(check_images=False), load_bins()


def target_matrix(samples: list[Sample]) -> Tensor:
    """[N, len(TARGETS)] float32, columns in TARGETS order."""
    return torch.tensor([[s.targets[t] for t in TARGETS] for s in samples], dtype=torch.float32)


def bin_index_matrix(samples: list[Sample], bins: dict[str, MacroBins]) -> Tensor:
    """[N, len(TARGETS)] long: the hard bin label per target."""
    return torch.tensor(
        [[bins[t].index_of(s.targets[t]) for t in TARGETS] for s in samples], dtype=torch.long
    )


def hl_gauss_sigma(b: MacroBins, sigma_bins: float) -> float:
    """sigma in log space, as a multiple of one (equal-width) bin."""
    return sigma_bins * (math.log(b.edges[-1] + b.offset) - math.log(b.offset)) / b.n_bins


def soft_targets(y: Tensor, bins: dict[str, MacroBins], sigma_bins: float = 0.75) -> list[Tensor]:
    """Per target, [N, K] HL-Gauss soft labels (`posterior.hl_gauss_target`)."""
    out = []
    for j, t in enumerate(TARGETS):
        b = bins[t]
        edges = torch.tensor(b.edges, dtype=torch.float32)
        out.append(hl_gauss_target(y[:, j], edges, b.offset, hl_gauss_sigma(b, sigma_bins)))
    return out


def soft_ce(logits: Tensor, target: Tensor) -> Tensor:
    return -(target * torch.log_softmax(logits.float(), -1)).sum(-1).mean()


def group_holdout(samples: list[Sample], fraction: float, seed: int) -> set[str]:
    """Ids of a held-out slice of `samples`, whole leakage groups at a time, per dataset.

    For a method that needs its own dev split carved from train (e.g. to select an
    epoch), so no group straddles train and dev.
    """
    rng = random.Random(seed)
    out: set[str] = set()
    for d in sorted({s.dataset for s in samples}):
        groups = sorted({s.group for s in samples if s.dataset == d})
        rng.shuffle(groups)
        n_d = sum(s.dataset == d for s in samples)
        chosen: set[str] = set()
        size = 0
        for g in groups:
            if size >= n_d * fraction:
                break
            chosen.add(g)
            size += sum(s.dataset == d and s.group == g for s in samples)
        out |= {s.id for s in samples if s.dataset == d and s.group in chosen}
    return out
