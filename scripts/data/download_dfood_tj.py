"""Download DFood-TJ-F-Nutr (4.2 GB) and unpack it. See docs/datasets.md.

Science Data Bank serves the file without login, but slowly per connection (~1 MB/s from
outside China), so it is fetched in parallel byte ranges and each part resumes on retry.
Only the F-Nutr part is needed: `DFoodTJ-Rec.zip` (19 GB) is dish names only.

**Patient photos** (dataset licence CC0 1.0, see NOTICE): never commit or share an image.

    uv run python scripts/data/download_dfood_tj.py [--out data/raw/dfood_tj] [--parts 16]
"""

from __future__ import annotations

import argparse
import os
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://download.scidb.cn/download?fileId="
FILES = {
    "DFoodTJ-F-Nutr.zip": ("cb2751a22a516c057913e15b7d3cc5c4", 4247942139),
    # the authors' baseline script, kept for reference
    "Nutr.py": ("ea4eb46ad7babec0c4831b75b505ae47", 13630),
}


def url_for(name: str) -> str:
    fid, _ = FILES[name]
    return f"{BASE}{fid}&path=/V3/{name}&fileName={name}"


def fetch_ranges(url: str, out: Path, size: int, parts: int) -> None:
    step = -(-size // parts)
    ranges = [(i, i * step, min(size, (i + 1) * step) - 1) for i in range(parts)]

    def get(r: tuple[int, int, int]) -> None:
        i, a, b = r
        p = Path(f"{out}.part{i:02d}")
        for attempt in range(50):
            have = p.stat().st_size if p.exists() else 0
            if have == b - a + 1:
                return
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={a + have}-{b}"})
                with urllib.request.urlopen(req, timeout=60) as resp, open(p, "ab") as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
            except OSError as e:
                print(f"  part {i} attempt {attempt}: {e}", flush=True)
                time.sleep(3)
        raise RuntimeError(f"part {i} failed after 50 attempts")

    with ThreadPoolExecutor(parts) as ex:
        list(ex.map(get, ranges))
    with open(out, "wb") as f:
        for i, _, _ in ranges:
            with open(f"{out}.part{i:02d}", "rb") as g:
                while chunk := g.read(1 << 24):
                    f.write(chunk)
    if out.stat().st_size != size:
        raise RuntimeError(f"{out}: size {out.stat().st_size} != {size}")
    for i, _, _ in ranges:
        os.remove(f"{out}.part{i:02d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/dfood_tj"))
    ap.add_argument("--parts", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, (_, size) in FILES.items():
        dest = args.out / name
        if dest.exists() and dest.stat().st_size == size:
            print(f"skip    {name}")
            continue
        t0 = time.time()
        print(f"fetch   {name} ({size / 2**30:.2f} GiB)", flush=True)
        fetch_ranges(url_for(name), dest, size, args.parts if size > 1 << 26 else 1)
        print(f"ok      {name} in {time.time() - t0:.0f}s")

    target = args.out / "DFoodTJ-F-Nutr"
    if (target / "truth" / "truth_raw.xlsx").exists():
        print("already unpacked")
        return
    with zipfile.ZipFile(args.out / "DFoodTJ-F-Nutr.zip") as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt member {bad}")
        z.extractall(args.out)
    n = len(list((target / "img").glob("*.png")))
    print(f"unpacked {n} images to {target}")
    if n != 3280:
        raise RuntimeError(f"expected 3280 images, found {n}")


if __name__ == "__main__":
    main()
