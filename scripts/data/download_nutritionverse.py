"""Download NutritionVerse-Real (1.1 GB) from Kaggle.

Needs a Kaggle API token (~/.kaggle/access_token or kaggle.json). The Kaggle CLI is run
through `uvx`, so it is never installed into the project venv. Licence: CC BY-NC-SA 4.0.

    uv run python scripts/data/download_nutritionverse.py
"""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

REAL = "nutritionverse/nutritionverse-real"


def kaggle_download(ref: str, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / (ref.split("/")[1] + ".zip")
    # The Kaggle CLI leaves `<zip>.kaggle-partial` next to an unfinished download; only a
    # zip without it is complete. Re-running the CLI resumes the partial file.
    partial = zip_path.with_name(zip_path.name + ".kaggle-partial")
    if zip_path.exists() and not partial.exists():
        print(f"skip    {ref}")
    else:
        print(f"fetch   {ref}", flush=True)
        subprocess.run(["uvx", "kaggle", "datasets", "download", ref, "-p", str(out)], check=True)
    return zip_path


def unzip_once(zip_path: Path, dest: Path, marker: Path, members: list[str] | None = None) -> None:
    if marker.exists():
        print(f"already unpacked {zip_path.name}")
        return
    with zipfile.ZipFile(zip_path) as z:
        names = members or z.namelist()
        if members:
            names = [n for n in z.namelist() if any(n.startswith(m) for m in members)]
        z.extractall(dest, names)
    print(f"unpacked {zip_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    args = ap.parse_args()
    real = args.root / "nutritionverse_real"
    z = kaggle_download(REAL, real)
    unzip_once(z, real, real / "nutritionverse_dish_metadata3.csv")


if __name__ == "__main__":
    main()
