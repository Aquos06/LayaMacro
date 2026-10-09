"""Download the benchmark-only datasets (never used for training). Resumable.

- `fpb`: Food Portion Benchmark (ISSAI, IEEE Access 2025), Hugging Face
  `issai/Food_Portion_Benchmark`, CC BY-NC 4.0. 138 Central Asian food classes, lab photos,
  **weighed grams per item** (6th column of the YOLO label files). Test labels are withheld
  by the authors, so only train/val are used. Model weights are skipped.
- `lefood`: LeFood-Set (Mendeley cchsk79jkt, 2025), CC BY 4.0. 524 Indonesian before/after
  pairs, **weighed on a digital scale** before and after eating, food names, no macros.

    uv run python scripts/data/download_benchmark_datasets.py fpb lefood
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path("data/raw")
#: Mendeley's file host answers 403 to urllib's default user agent.
UA = {"User-Agent": "Mozilla/5.0 (layamacro dataset downloader)"}
LEFOOD_API = "https://data.mendeley.com/public-api/datasets/cchsk79jkt"


def fetch(url: str, dest: Path) -> None:
    """Download to dest. Mendeley sometimes answers with a small JSON {"url": ...} pointing at
    the real file instead of redirecting; follow it rather than saving the JSON as the image."""
    for _ in range(3):
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if data[:8] == b'{"url":"':
            url = json.loads(data)["url"]
            continue
        dest.write_bytes(data)
        return
    raise RuntimeError(f"{dest.name}: too many JSON redirects")


def fpb() -> None:
    from huggingface_hub import snapshot_download

    out = ROOT / "food_portion_benchmark"
    snapshot_download(
        "issai/Food_Portion_Benchmark",
        repo_type="dataset",
        local_dir=out,
        allow_patterns=["FPB_Dataset/*", "README.md"],
    )
    n = len(list(out.glob("FPB_Dataset/RGB/*/images/*")))
    print(f"fpb: {n} images in {out}")


def lefood() -> None:
    out = ROOT / "lefood_set"
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(LEFOOD_API, headers=UA), timeout=60) as r:
        meta = json.load(r)
    files = meta["files"]
    for f in files:
        dest = out / f["filename"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            continue
        fetch(f["content_details"]["download_url"], dest)
    print(f"lefood: {len(files)} files in {out} (licence {meta['data_licence']['short_name']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+", choices=("fpb", "lefood"))
    args = ap.parse_args()
    for w in args.which:
        {"fpb": fpb, "lefood": lefood}[w]()


if __name__ == "__main__":
    main()
