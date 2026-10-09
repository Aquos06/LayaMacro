"""Image loading shared by every encoder and model."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def load_image(path: Path) -> Image.Image:
    """RGB with EXIF rotation applied: phone photos (DFood-TJ, NutritionVerse) carry it.

    4032x3024 phone JPEGs are decoded at reduced size (`draft`), which every model then
    downsamples further anyway.
    """
    img = Image.open(path)
    if img.format == "JPEG":
        img.draft("RGB", (1024, 1024))
    return ImageOps.exif_transpose(img).convert("RGB")
