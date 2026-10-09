"""Train MacroPosterior: SigLIP 2 patches -> attention pool -> 18-bin distribution per target.

The best LayaMacro calorie model (Nutrition5k kcal MAE 41.4, 2 seeds) trains the last 12 of
SigLIP 2's 24 blocks together with the head: `--tune-last 12`. The frozen first blocks are run
once by `scripts/data/cache_features.py --hidden-layer 12`, so training uses that cache.

Recipe: 40 epochs, AdamW (head 3e-4, encoder blocks 1e-5), OneCycle, HL-Gauss soft labels
(sigma 0.75 bins), batch 64, no checkpoint selection. Writes runs/<name>/ (predictions on the
calibration + test photos) and the weights (`model.pt`).

    uv run python scripts/train/train_macroposterior.py --tune-last 12 --seed 0
    uv run python scripts/train/train_macroposterior.py --seed 0          # frozen encoder
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from layamacro.eval.evaluate import eval_samples, save_predictions
from layamacro.posterior import MacroPosterior
from layamacro.schema import TARGETS
from layamacro.train.common import (
    load_data,
    save_weights,
    seed_everything,
    soft_ce,
    soft_targets,
    target_matrix,
)
from layamacro.vision.tail import SiglipTail


class RamStore:
    """Features kept in RAM, gathered per batch onto the GPU."""

    def __init__(self, feats: torch.Tensor, dev: torch.device) -> None:
        self.feats, self.dev = feats, dev

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        return self.feats[idx.cpu()].to(self.dev, non_blocking=True)


class Tuned(torch.nn.Module):
    """Trainable SigLIP 2 tail, then MacroPosterior: same interface as MacroPosterior."""

    def __init__(self, tail: SiglipTail, head: MacroPosterior) -> None:
        super().__init__()
        self.tail, self.head = tail, head

    def forward(self, hidden: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            patches = self.tail(hidden)
        return self.head(patches)


def fit(
    model: MacroPosterior | Tuned,
    feats: torch.Tensor | RamStore,
    idx: torch.Tensor,
    soft: list[torch.Tensor],
    args: argparse.Namespace,
) -> None:
    """AdamW + OneCycle over `args.epochs`. soft[j] rows align with idx."""
    if isinstance(model, Tuned):
        groups = [
            {"params": model.tail.parameters(), "lr": args.lr_encoder},
            {"params": model.head.parameters(), "lr": args.lr},
        ]
        max_lr: float | list[float] = [args.lr_encoder, args.lr]
    else:
        groups = [{"params": model.parameters(), "lr": args.lr}]
        max_lr = args.lr
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(idx) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr, total_steps=steps, pct_start=0.05)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(idx), device=idx.device)
        tot = 0.0
        for k in range(0, len(order), args.batch):
            b = order[k : k + args.batch]
            logits, _ = model(feats[idx[b]])
            loss = sum(soft_ce(lg, soft[j][b]) for j, lg in enumerate(logits))
            opt.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[union-attr]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(b)
        if epoch % 10 == 9 or epoch == args.epochs - 1:
            print(f"epoch {epoch + 1}  loss {tot / len(idx):.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/features/siglip2.pt"))
    ap.add_argument("--tune-last", type=int, default=0, help="also train SigLIP 2's last k blocks")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--bench",
        action="store_true",
        help="also predict the unseen-dataset benchmark photos (FPB, LeFood-Set)",
    )
    ap.add_argument("--bench-features", type=Path, default=Path("data/features/siglip2-bench.pt"))
    ap.add_argument("--tag", default="", help="run-name suffix")
    args = ap.parse_args()
    seed_everything(args.seed)
    dev = torch.device(args.device)

    samples, bins = load_data()
    model: MacroPosterior | Tuned = MacroPosterior(bins, d_in=1024, targets=TARGETS).to(dev)
    feats: torch.Tensor | RamStore
    if args.tune_last:
        tail = SiglipTail(args.tune_last).to(dev)
        model = Tuned(tail, model)  # type: ignore[arg-type]
        hidden = args.features.with_name(
            f"{args.features.stem}-L{tail.first}{args.features.suffix}"
        )
        cache = torch.load(hidden, weights_only=False)
        assert cache["hidden_layer"] == tail.first, (hidden, cache["hidden_layer"])
        final = torch.load(args.features, weights_only=False)
        assert final["ids"] == cache["ids"], "hidden and final caches are not row-aligned"
        err = tail.check_against(cache["features"][:16], final["features"][:16])
        print(f"tail reproduces the frozen encoder: rel err {err:.4f}", flush=True)
        del final
        feats = RamStore(cache["features"].pin_memory(), dev)
    else:
        cache = torch.load(args.features, weights_only=False)
        feats = cache["features"].to(dev)
    row = {i: k for k, i in enumerate(cache["ids"])}
    n_params = sum(p.numel() for p in model.parameters())

    train = [s for s in samples if s.split == "train"]
    tr_idx = torch.tensor([row[s.id] for s in train], device=dev)
    tr_soft = [x.to(dev) for x in soft_targets(target_matrix(train), bins)]
    t0 = time.time()
    fit(model, feats, tr_idx, tr_soft, args)
    train_s = time.time() - t0

    ev = eval_samples(samples)
    idx = torch.tensor([row[s.id] for s in ev], device=dev)
    model.eval()
    outs: list[list[np.ndarray]] = [[] for _ in TARGETS]
    with torch.no_grad():
        for k in range(0, len(idx), 256):
            lg, _ = model(feats[idx[k : k + 256]])
            for j, x in enumerate(lg):
                outs[j].append(x.float().cpu().numpy())
    name = "macroposterior" + (f"_ft{args.tune_last}" if args.tune_last else "")
    run = save_predictions(
        name + (f"_{args.tag}" if args.tag else "") + f"_s{args.seed}",
        [s.id for s in ev],
        {t: np.concatenate(outs[j]) for j, t in enumerate(TARGETS)},
        {
            "method": "MacroPosterior: SigLIP 2-large"
            + (f", last {args.tune_last} blocks trained" if args.tune_last else " (frozen)")
            + " -> attention pool (1 query per target) -> MLP -> 18-bin softmax per target",
            "trained": f"{args.epochs} epochs, AdamW {args.lr} (encoder {args.lr_encoder}), "
            f"batch {args.batch}, seed {args.seed}, {len(train)} train photos",
            "params_trainable": n_params,
            "train_seconds": round(train_s),
            "encoder": cache["model_id"],
        },
    )
    ckpt = save_weights(run, {"model": model.state_dict(), "args": vars(args)})
    print(f"wrote {run} ({n_params / 1e6:.2f}M params, {train_s:.0f}s); weights {ckpt}")

    if args.bench:
        from layamacro.eval.bench import bench_samples

        del feats, cache
        bpath = args.bench_features
        if args.tune_last:
            bpath = bpath.with_name(f"{bpath.stem}-L{tail.first}{bpath.suffix}")
        bc = torch.load(bpath, weights_only=False)
        assert bc.get("hidden_layer") == (tail.first if args.tune_last else None), bpath
        brow = {i: k for k, i in enumerate(bc["ids"])}
        bs = bench_samples()
        bfeats = RamStore(bc["features"], dev)
        bidx = torch.tensor([brow[s.id] for s in bs])
        bouts: list[list[np.ndarray]] = [[] for _ in TARGETS]
        with torch.no_grad():
            for k in range(0, len(bidx), 128):
                lg, _ = model(bfeats[bidx[k : k + 128]])
                for j, x in enumerate(lg):
                    bouts[j].append(x.float().cpu().numpy())
        arrays = {"ids": np.array([s.id for s in bs])} | {
            f"logits_{t}": np.concatenate(bouts[j]) for j, t in enumerate(TARGETS)
        }
        np.savez(Path(run) / "bench.npz", **arrays)  # type: ignore[arg-type]
        print(f"wrote {run}/bench.npz ({len(bs)} benchmark photos)")


if __name__ == "__main__":
    main()
