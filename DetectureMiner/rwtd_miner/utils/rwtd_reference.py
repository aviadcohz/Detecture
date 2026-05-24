from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from rwtd_miner.stages.stage_a_maskstats import _stage_a_rwtd_score
from rwtd_miner.utils.io import ensure_dir
from rwtd_miner.utils.sa1b_geometry import (
    _balance_score,
    _boundary_pixels_and_counts,
    _compute_texture_boundary_score,
    _render_assets,
)

# GitHub coordinates for the upstream TextureSAM reference release.
# Withheld from this submission archive for double-blind review;
# will be restored in the public version upon acceptance.
_GITHUB_OWNER = "<withheld-for-review>"
_GITHUB_REPO = "TextureSAM"
_GITHUB_BRANCH = "main"


def _github_json(path: str, retries: int = 3, timeout: float = 30.0) -> Any:
    url = f"https://api.github.com/repos/{_GITHUB_OWNER}/{_GITHUB_REPO}/contents/{path}?ref={_GITHUB_BRANCH}"
    last_err: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rwtd-miner"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover
            last_err = exc
            time.sleep(0.8)
    if last_err:
        raise last_err
    raise RuntimeError(f"Failed to fetch GitHub JSON for {path}")


def _download_file(url: str, dst: Path, retries: int = 3, timeout: float = 60.0) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    ensure_dir(dst.parent)
    last_err: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rwtd-miner"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            dst.write_bytes(data)
            if dst.stat().st_size <= 0:
                raise RuntimeError(f"Downloaded empty file: {dst}")
            return
        except Exception as exc:  # pragma: no cover
            last_err = exc
            if dst.exists() and dst.stat().st_size == 0:
                dst.unlink(missing_ok=True)
            time.sleep(0.8)
    if last_err:
        raise last_err


def _numeric_stem_key(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    if stem.isdigit():
        return int(stem), name
    return 10**9, name


def fetch_rwtd_from_texturesam(out_root: Path, max_images: int | None = None) -> dict[str, Any]:
    images_dir = ensure_dir(out_root / "images")
    labels_dir = ensure_dir(out_root / "labels")

    image_items = _github_json("Kaust256/images")
    label_items = _github_json("Kaust256/labeles")
    if not isinstance(image_items, list) or not isinstance(label_items, list):
        raise RuntimeError("Unexpected GitHub API schema while listing Kaust256")

    label_by_stem: dict[str, dict[str, Any]] = {}
    for item in label_items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        if not name.lower().endswith(".png"):
            continue
        label_by_stem[Path(name).stem] = item

    valid_images: list[dict[str, Any]] = []
    for item in image_items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        if Path(name).stem not in label_by_stem:
            continue
        valid_images.append(item)

    valid_images = sorted(valid_images, key=lambda x: _numeric_stem_key(str(x.get("name", ""))))
    if max_images is not None and max_images > 0:
        valid_images = valid_images[: int(max_images)]

    downloaded = 0
    reused = 0
    pairs = 0
    for item in tqdm(valid_images, desc="download_rwtd", unit="img"):
        image_name = str(item["name"])
        stem = Path(image_name).stem
        label_item = label_by_stem.get(stem)
        if label_item is None:
            continue

        image_dst = images_dir / image_name
        label_dst = labels_dir / str(label_item["name"])

        before_i = image_dst.exists() and image_dst.stat().st_size > 0
        before_l = label_dst.exists() and label_dst.stat().st_size > 0

        _download_file(str(item["download_url"]), image_dst)
        _download_file(str(label_item["download_url"]), label_dst)

        if before_i and before_l:
            reused += 1
        else:
            downloaded += 1
        pairs += 1

    return {
        "source": f"https://github.com/{_GITHUB_OWNER}/{_GITHUB_REPO}",
        "dataset": "Kaust256 (RWTD)",
        "pairs_requested": len(valid_images),
        "pairs_downloaded_new": downloaded,
        "pairs_reused_existing": reused,
        "pairs_ready": pairs,
        "images_dir": str(images_dir.resolve()),
        "labels_dir": str(labels_dir.resolve()),
    }


def _entropy(ratios: np.ndarray) -> float:
    if ratios.size == 0:
        return 0.0
    p = ratios / max(1e-9, float(ratios.sum()))
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _analyze_rwtd_pair(
    *,
    image_id: str,
    image_path: Path,
    label_path: Path,
    cache_meta_path: Path,
    mask_out: Path,
    overlay_out: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    if cache_meta_path.exists() and mask_out.exists() and overlay_out.exists():
        try:
            return json.loads(cache_meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
    with Image.open(label_path) as lm:
        lbl = lm.convert("L")
        if lbl.size != (w, h):
            lbl = lbl.resize((w, h), Image.Resampling.NEAREST)
        lab = np.asarray(lbl)

    thr = int(cfg.get("rwtd_label_threshold", 127))
    cls1 = lab > thr
    cls0 = ~cls1
    region_map = np.zeros((h, w), dtype=np.int32)

    areas: list[int] = []
    rid = 1
    structure = np.ones((3, 3), dtype=np.uint8)
    for c in (cls0, cls1):
        cc, ncc = ndimage.label(c.astype(np.uint8), structure=structure)
        for cid in range(1, int(ncc) + 1):
            m = cc == cid
            area = int(m.sum())
            if area <= 0:
                continue
            region_map[m] = rid
            areas.append(area)
            rid += 1

    total = float(h * w)
    ratios = np.asarray(areas, dtype=np.float64) / max(1.0, total) if areas else np.asarray([], dtype=np.float64)

    n_masks = int(len(areas))
    largest_ratio = float(ratios.max()) if ratios.size else 0.0
    median_ratio = float(np.median(ratios)) if ratios.size else 0.0
    small_threshold = float(cfg.get("small_threshold", 0.001))
    small_frac = float((ratios < small_threshold).mean()) if ratios.size else 0.0
    area_entropy = _entropy(ratios)
    stage_a_score = _stage_a_rwtd_score(
        n_masks=n_masks,
        largest_ratio=largest_ratio,
        median_ratio=median_ratio,
        small_frac=small_frac,
        entropy=area_entropy,
    )

    min_large_region_frac = float(cfg.get("min_large_region_frac", 0.08))
    min_large_area = min_large_region_frac * total
    large_ids = {i + 1 for i, a in enumerate(areas) if float(a) >= min_large_area}
    boundary_pixels, pair_counts = _boundary_pixels_and_counts(region_map, large_ids=large_ids)
    strong_min = int(cfg.get("strong_boundary_min_pixels", 40))
    strong_count = int(sum(1 for v in pair_counts.values() if int(v) >= strong_min))
    perimeter = float(2 * (h + w))
    boundary_norm = float(sum(pair_counts.values())) / max(1.0, perimeter)
    large_areas = [int(areas[i - 1]) for i in sorted(large_ids) if 0 <= (i - 1) < len(areas)]
    balance = _balance_score(large_areas)
    texture_boundary_score = _compute_texture_boundary_score(
        large_count=len(large_ids),
        strong_count=strong_count,
        boundary_norm=boundary_norm,
        balance=balance,
        object_fraction=0.0,
    )

    raw_mask = np.where(region_map > 0, 1, 0).astype(np.uint8)
    _render_assets(
        image_path=image_path,
        raw_mask=raw_mask,
        texture_region_map=region_map.astype(np.int32),
        boundary_pixels=boundary_pixels,
        mask_out=mask_out,
        overlay_out=overlay_out,
    )

    result = {
        "image_id": image_id,
        "stageA_status": "ok",
        "stageA_n_masks": n_masks,
        "stageA_largest_ratio": float(largest_ratio),
        "stageA_median_ratio": float(median_ratio),
        "stageA_small_frac": float(small_frac),
        "stageA_area_entropy": float(area_entropy),
        "stageA_rwtd_score": float(stage_a_score),
        "stageA_pass": bool(len(large_ids) >= 2 and texture_boundary_score >= 55.0),
        "stageA_error": None,
        "geom_status": "ok",
        "geom_texture_boundary_score": float(texture_boundary_score),
        "geom_object_fraction": 0.0,
        "geom_texture_fraction": 1.0,
        "geom_ambiguous_fraction": 0.0,
        "geom_num_large_texture_regions": int(len(large_ids)),
        "geom_num_strong_boundaries": int(strong_count),
        "geom_boundary_norm": float(boundary_norm),
        "geom_mask_rel": str(mask_out.resolve()),
        "geom_overlay_rel": str(overlay_out.resolve()),
    }
    ensure_dir(cache_meta_path.parent)
    cache_meta_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def score_rwtd_reference_dataset(
    *,
    rwtd_root: Path,
    batch_dir: Path,
    cfg: dict[str, Any],
    max_images: int | None = None,
) -> pd.DataFrame:
    images_dir = rwtd_root / "images"
    labels_dir = rwtd_root / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"RWTD images/labels folders not found under: {rwtd_root}")

    masks_dir = ensure_dir(batch_dir / "review" / "masks")
    overlays_dir = ensure_dir(batch_dir / "review" / "overlays")
    meta_dir = ensure_dir(batch_dir / "review" / "meta_rwtd")

    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
        key=lambda p: _numeric_stem_key(p.name),
    )
    if max_images is not None and max_images > 0:
        image_paths = image_paths[: int(max_images)]

    rows: list[dict[str, Any]] = []
    for image_path in tqdm(image_paths, desc="score_rwtd", unit="img"):
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.png"
        if not label_path.exists():
            continue
        image_id = f"rwtd_{stem}"
        base = {
            "image_id": image_id,
            "source_image_id": stem,
            "image_path": str(image_path.resolve()),
            "annotation_ref": str(label_path.resolve()),
            "file_size_bytes": int(image_path.stat().st_size),
            "dataset": "rwtd_reference",
            "resolution_ok": True,
            "batch_id": 0,
            "stageB_pos_score": np.nan,
            "stageB_neg_score": np.nan,
            "stageB_clip_score": np.nan,
            "stageB_rank": np.nan,
            "stageB_pass": np.nan,
            "stageB_error": "not_run",
            "stageC_caption": None,
            "stageC_pass": None,
            "stageD_score_0_100": None,
            "stageD_decision": None,
            "stageD_flags": None,
            "stageD_reason": None,
        }

        with Image.open(image_path) as im:
            w, h = im.size
        base["width"] = int(w)
        base["height"] = int(h)

        analyzed = _analyze_rwtd_pair(
            image_id=image_id,
            image_path=image_path,
            label_path=label_path,
            cache_meta_path=meta_dir / f"{image_id}.json",
            mask_out=masks_dir / f"{image_id}.jpg",
            overlay_out=overlays_dir / f"{image_id}.jpg",
            cfg=cfg,
        )
        base.update(analyzed)
        rows.append(base)

    return pd.DataFrame(rows)


def apply_texture_priority_scoring(
    df: pd.DataFrame,
    selected_min: float = 74.0,
    borderline_min: float = 62.0,
) -> pd.DataFrame:
    out = df.copy()
    stage_a = out.get("stageA_rwtd_score", pd.Series([np.nan] * len(out))).fillna(35.0).astype(float)
    geom = out.get("geom_texture_boundary_score", pd.Series([np.nan] * len(out))).fillna(0.0).astype(float)
    obj = out.get("geom_object_fraction", pd.Series([np.nan] * len(out))).fillna(0.0).astype(float)
    tex = out.get("geom_texture_fraction", pd.Series([np.nan] * len(out))).fillna(0.0).astype(float)
    amb = out.get("geom_ambiguous_fraction", pd.Series([np.nan] * len(out))).fillna(0.0).astype(float)
    large = out.get("geom_num_large_texture_regions", pd.Series([np.nan] * len(out))).fillna(0.0).astype(float)
    strong = out.get("geom_num_strong_boundaries", pd.Series([np.nan] * len(out))).fillna(0.0).astype(float)

    tex_bonus = 20.0 * np.clip((tex - 0.65) / 0.35, 0.0, 1.0)
    region_bonus = np.where((large >= 2.0) & (large <= 4.0), 6.0, np.where(large >= 2.0, 2.0, -8.0))
    strong_bonus = np.where(strong >= 1.0, 3.0, -4.0)
    obj_penalty = 180.0 * np.clip(obj - 0.002, 0.0, 1.0)
    amb_penalty = 120.0 * np.clip(amb, 0.0, 1.0)
    clutter_penalty = 4.0 * np.clip(large - 5.0, 0.0, 99.0)

    score = (0.20 * stage_a) + (0.60 * geom) + tex_bonus + region_bonus + strong_bonus - obj_penalty - amb_penalty - clutter_penalty
    score = np.clip(score, 0.0, 100.0)

    out["review_score"] = score
    out["selection_reason"] = "texture_priority_calibration"
    out["final_selected"] = score >= float(selected_min)
    out["final_borderline"] = (~out["final_selected"]) & (score >= float(borderline_min))
    return out
