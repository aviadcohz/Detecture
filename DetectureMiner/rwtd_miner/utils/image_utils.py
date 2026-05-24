from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps, ImageDraw


def safe_image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as im:
            w, h = im.size
        return int(w), int(h)
    except Exception:
        return None, None


def load_rgb(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None


def make_contact_sheet(
    image_paths: Iterable[Path],
    out_path: Path,
    thumb_size: tuple[int, int] = (224, 224),
    cols: int = 10,
    title: str | None = None,
) -> None:
    paths = [p for p in image_paths if p.exists()]
    if not paths:
        return

    rows = math.ceil(len(paths) / cols)
    top_pad = 40 if title else 0
    canvas = Image.new("RGB", (cols * thumb_size[0], rows * thumb_size[1] + top_pad), (245, 247, 250))

    if title:
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 10), title, fill=(20, 30, 40))

    for idx, p in enumerate(paths):
        im = load_rgb(p)
        if im is None:
            continue
        tile = ImageOps.fit(im, thumb_size, method=Image.Resampling.BICUBIC)
        x = (idx % cols) * thumb_size[0]
        y = (idx // cols) * thumb_size[1] + top_pad
        canvas.paste(tile, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def sample_paths(paths: list[Path], count: int, seed: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    rnd = random.Random(seed)
    return rnd.sample(paths, count)


def score_histogram(scores: list[float], out_path: Path) -> None:
    if not scores:
        return
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(scores, bins=20, color="#3a78b5", edgecolor="#1f3d5e", alpha=0.85)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("RWTD Score Histogram")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except Exception:
        im = load_rgb(src)
        if im is not None:
            im.save(dst, quality=90)
