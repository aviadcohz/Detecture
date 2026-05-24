from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import fsspec
from PIL import Image

from rwtd_miner.utils.io import ensure_dir
from rwtd_miner.utils.logging import get_logger


def _resize_rgb_bytes(raw: bytes, target_long_side: int) -> bytes:
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        if target_long_side > 0:
            w, h = im.size
            long_side = max(w, h)
            if long_side > target_long_side:
                scale = target_long_side / float(long_side)
                nw = max(1, int(round(w * scale)))
                nh = max(1, int(round(h * scale)))
                im = im.resize((nw, nh), Image.Resampling.BICUBIC)
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=90)
        return out.getvalue()


def fetch_sa1b_pairs_from_tar(
    part_url: str,
    out_root: Path,
    num_images: int = 100,
    image_target_long_side: int = 1024,
) -> dict:
    log = get_logger("sa1b_sample_fetch")
    images_dir = ensure_dir(out_root / "images")
    ann_dir = ensure_dir(out_root / "annotations")

    has_img: set[str] = set()
    has_ann: set[str] = set()
    paired: list[str] = []

    # Existing files are reused to allow resume.
    for p in images_dir.glob("*.jpg"):
        has_img.add(p.stem)
    for p in ann_dir.glob("*.json"):
        has_ann.add(p.stem)
    for stem in sorted(has_img & has_ann):
        paired.append(stem)
    if len(paired) >= num_images:
        return {
            "part_url": part_url,
            "target": int(num_images),
            "paired": int(len(paired)),
            "images_dir": str(images_dir),
            "annotations_dir": str(ann_dir),
            "reused_existing": True,
        }

    with fsspec.open(part_url, "rb").open() as fobj:
        tf = tarfile.open(fileobj=fobj, mode="r|*")
        for member in tf:
            if not member.isfile():
                continue
            name = str(member.name)
            low = name.lower()
            if not (low.endswith(".jpg") or low.endswith(".jpeg") or low.endswith(".json")):
                continue

            stem = Path(name).stem
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            extracted.close()

            if low.endswith(".json"):
                out_path = ann_dir / f"{stem}.json"
                if not out_path.exists():
                    try:
                        # Normalize JSON so local adapter has one-annotation-per-image path.
                        parsed = json.loads(payload)
                        out_path.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
                    except Exception:
                        continue
                has_ann.add(stem)
            else:
                out_path = images_dir / f"{stem}.jpg"
                if not out_path.exists():
                    try:
                        resized = _resize_rgb_bytes(payload, target_long_side=image_target_long_side)
                    except Exception:
                        continue
                    out_path.write_bytes(resized)
                has_img.add(stem)

            if stem in has_img and stem in has_ann and stem not in paired:
                paired.append(stem)
                if len(paired) % 10 == 0:
                    log.info("Fetched %s/%s paired samples", len(paired), num_images)
                if len(paired) >= num_images:
                    break

    keep = set(paired[:num_images])

    # Prune extras from this sampling root for deterministic exactly-N pilot set.
    for p in images_dir.glob("*.jpg"):
        if p.stem not in keep:
            p.unlink()
    for p in ann_dir.glob("*.json"):
        if p.stem not in keep:
            p.unlink()

    return {
        "part_url": part_url,
        "target": int(num_images),
        "paired": int(len(keep)),
        "images_dir": str(images_dir),
        "annotations_dir": str(ann_dir),
        "reused_existing": False,
    }
