"""Encode every photo once with SigLIP 2 and cache the patch tokens for training.

The first SigLIP 2 blocks never train, so their output is computed once:

- default: the final patch tokens, fp16 [N, 576, 1024] -> data/features/siglip2.pt. Feeds
  the frozen-encoder MacroPosterior and Laya graft.
- `--hidden-layer N`: the hidden state *entering* block N (`hidden_states[N]` of the HF
  model; index 0 is the patch embeddings), so the last 24 - N blocks can be trained on top
  of it -> siglip2-LN.pt. The released models use N = 12 (MacroPosterior) and N = 20 (Laya).
- `--bench`: the same for the unseen-dataset benchmark photos (FPB, LeFood-Set) ->
  siglip2-bench[-LN].pt.

Phone photos carry EXIF rotation, so it is applied before encoding.

    uv run python scripts/data/cache_features.py                 # final tokens
    uv run python scripts/data/cache_features.py --hidden-layer 12
    uv run python scripts/data/cache_features.py --bench --hidden-layer 12
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from layamacro.data.images import load_image
from layamacro.data.samples import BENCH_DATASETS, load_samples
from layamacro.vision.encoder import SIGLIP2_LARGE_384, FrozenVisionEncoder


class Images(Dataset):
    def __init__(self, paths: list[Path], processor: Any) -> None:
        self.paths, self.processor = paths, processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        px = self.processor(images=[load_image(self.paths[i])], return_tensors="pt")
        return px["pixel_values"][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/features"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bench", action="store_true", help="encode the benchmark-only datasets")
    ap.add_argument(
        "--hidden-layer",
        type=int,
        default=None,
        help="cache the input to this block (0-24) instead of the final output",
    )
    args = ap.parse_args()

    samples = load_samples(BENCH_DATASETS) if args.bench else load_samples()
    ids = [s.id for s in samples]
    paths = [s.image for s in samples]
    print(f"{len(ids)} photos ({'benchmark' if args.bench else 'training/evaluation'} datasets)")

    from transformers import AutoImageProcessor

    enc = FrozenVisionEncoder(SIGLIP2_LARGE_384).to(args.device, torch.bfloat16)
    proc = AutoImageProcessor.from_pretrained(SIGLIP2_LARGE_384)
    loader = DataLoader(
        Images(paths, proc), batch_size=args.batch, num_workers=args.workers, shuffle=False
    )
    # Filled in place: concatenating batches would briefly need twice the memory.
    features: torch.Tensor | None = None
    t0 = time.time()
    with torch.inference_mode():
        for k, px in enumerate(loader):
            px = px.to(args.device, torch.bfloat16)
            if args.hidden_layer is not None:
                hs = enc.model(pixel_values=px, output_hidden_states=True).hidden_states
                out = hs[args.hidden_layer].to(torch.float16)
            else:
                out = enc(px).to(torch.float16)
            if features is None:
                features = torch.empty((len(ids), *out.shape[1:]), dtype=out.dtype)
            start = k * args.batch
            features[start : start + len(out)] = out.cpu()
            if k % 50 == 0:
                print(f"  {min((k + 1) * args.batch, len(ids))}/{len(ids)}", flush=True)
    assert features is not None
    for start in range(0, len(features), 1024):  # chunked: no full-size temporary
        if not torch.isfinite(features[start : start + 1024]).all():
            raise RuntimeError("non-finite features")
    print(f"encoded {tuple(features.shape)} in {time.time() - t0:.0f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = ("-bench" if args.bench else "") + (
        f"-L{args.hidden_layer}" if args.hidden_layer is not None else ""
    )
    path = args.out / f"siglip2{suffix}.pt"
    torch.save(
        {
            "ids": ids,
            "features": features,
            "model_id": SIGLIP2_LARGE_384,
            "hidden_layer": args.hidden_layer,
        },
        path,
    )
    print(f"wrote {path} ({path.stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
