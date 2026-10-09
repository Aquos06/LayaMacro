"""Download the Nutrition5k subset we use: metadata, splits, overhead RGB only.

~1.35 GiB, not the 181 GiB tarball (docs/datasets.md). Anonymous HTTPS against
`storage.googleapis.com` — the README's `storage.cloud.google.com` 302s to a login page.

Only dishes in the *depth* splits are fetched: they are the overhead subset (the rgb
splits are side-angle video frames). Idempotent — existing files are skipped, so a partial
run resumes.

    uv run python scripts/data/download_nutrition5k.py [--out data/raw/nutrition5k] [--workers 16]
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/"

METADATA = (
    "metadata/dish_metadata_cafe1.csv",
    "metadata/dish_metadata_cafe2.csv",
    "metadata/ingredients_metadata.csv",
    "dish_ids/splits/depth_train_ids.txt",
    "dish_ids/splits/depth_test_ids.txt",
    "dish_ids/splits/rgb_train_ids.txt",
    "dish_ids/splits/rgb_test_ids.txt",
)


def fetch(rel: str, out: Path) -> str:
    """Fetch one file. Returns 'ok', 'skip' or 'missing'."""
    dest = out / rel
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(BASE + rel, timeout=60) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 16):
                f.write(chunk)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        if e.code == 404:
            return "missing"
        raise
    tmp.rename(dest)
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/raw/nutrition5k"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    for rel in METADATA:
        print(f"{fetch(rel, args.out):7s} {rel}")

    ids: list[str] = []
    for split in ("depth_train_ids.txt", "depth_test_ids.txt"):
        text = (args.out / "dish_ids/splits" / split).read_text()
        ids += [line.strip() for line in text.splitlines() if line.strip()]
    rels = [f"imagery/realsense_overhead/{d}/rgb.png" for d in sorted(set(ids))]
    print(f"fetching {len(rels)} overhead images with {args.workers} workers")

    counts = {"ok": 0, "skip": 0, "missing": 0}
    missing: list[str] = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch, r, args.out): r for r in rels}
        for i, fut in enumerate(as_completed(futures), 1):
            status = fut.result()
            counts[status] += 1
            if status == "missing":
                missing.append(futures[fut])
            if i % 250 == 0 or i == len(rels):
                print(f"  {i}/{len(rels)}  {counts}", flush=True)

    # Expected: exactly 3 depth_train ids have no overhead dir.
    print(f"done: {counts}")
    for m in sorted(missing):
        print(f"  no overhead image: {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
