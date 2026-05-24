from __future__ import annotations

import json
import math
import re
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from rwtd_miner.stages.stage_a_maskstats import _stage_a_rwtd_score
from rwtd_miner.utils.io import ensure_dir, write_json
from rwtd_miner.utils.rwtd_reference import apply_texture_priority_scoring
from rwtd_miner.utils.sa1b_geometry import (
    _balance_score,
    _boundary_pixels_and_counts,
    _compute_texture_boundary_score,
    _render_assets,
)

ADE20K_URLS = [
    "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
    "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
]

TEXTURE_KEYWORDS = (
    "road",
    "street",
    "pavement",
    "sidewalk",
    "curb",
    "floor",
    "wall",
    "tile",
    "brick",
    "stone",
    "rock",
    "concrete",
    "asphalt",
    "sky",
    "cloud",
    "water",
    "sea",
    "river",
    "sand",
    "soil",
    "dirt",
    "mud",
    "grass",
    "vegetation",
    "foliage",
    "tree",
    "plant",
    "ground",
    "earth",
    "mountain",
    "snow",
    "wood",
    "metal",
    "glass",
    "fabric",
    "carpet",
    "field",
)

OBJECT_KEYWORDS = (
    "person",
    "man",
    "woman",
    "child",
    "face",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "bike",
    "dog",
    "cat",
    "animal",
    "bird",
    "chair",
    "table",
    "sofa",
    "couch",
    "bed",
    "cabinet",
    "shelf",
    "door",
    "clock",
    "screen",
    "computer",
    "monitor",
    "tv",
    "phone",
    "bottle",
    "food",
    "plate",
    "toilet",
    "sink",
    "oven",
    "microwave",
    "vehicle",
)

AMBIGUOUS_KEYWORDS = (
    "building",
    "house",
    "structure",
    "window",
    "fence",
    "bridge",
    "tower",
    "sign",
    "pole",
    "light",
)

_STRUCT = np.ones((3, 3), dtype=np.uint8)


def _download_stream(url: str, dst: Path) -> None:
    ensure_dir(dst.parent)
    req = urllib.request.Request(url, headers={"User-Agent": "rwtd-miner"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0) or 0)
        with dst.open("wb") as f:
            with tqdm(total=total if total > 0 else None, unit="B", unit_scale=True, desc="download_ade20k") as bar:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))


def ensure_ade20k_dataset(root_dir: Path) -> dict[str, Any]:
    ensure_dir(root_dir)
    ade_dir = root_dir / "ADEChallengeData2016"
    ready = (ade_dir / "images" / "training").exists() and (ade_dir / "annotations" / "training").exists()
    if ready:
        return {"status": "ready", "ade_root": str(ade_dir.resolve()), "downloaded": False}

    zip_path = root_dir / "ADEChallengeData2016.zip"
    if not zip_path.exists() or zip_path.stat().st_size <= 0:
        last_err: Exception | None = None
        for url in ADE20K_URLS:
            try:
                _download_stream(url, zip_path)
                last_err = None
                break
            except Exception as exc:  # pragma: no cover
                last_err = exc
        if last_err:
            raise last_err

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root_dir)

    if not ready:
        ready = (ade_dir / "images" / "training").exists() and (ade_dir / "annotations" / "training").exists()
    if not ready:
        raise RuntimeError(f"ADE20K extraction failed or unexpected layout under: {root_dir}")

    return {"status": "downloaded", "ade_root": str(ade_dir.resolve()), "downloaded": True}


def _parse_object_info(path: Path) -> dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing ADE20K class file: {path}")

    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = re.split(r"\t+", s)
        idx: int | None = None
        name: str | None = None
        if parts and parts[0].isdigit():
            idx = int(parts[0])
            if len(parts) >= 5 and parts[4].strip():
                name = parts[4].strip()
            elif len(parts) >= 2 and parts[1].strip():
                name = parts[1].strip()
        else:
            m = re.match(r"^\s*(\d+)\s+(.+)$", s)
            if m:
                idx = int(m.group(1))
                name = m.group(2).strip()
        if idx is None or name is None:
            continue
        out[idx] = name.lower()
    if not out:
        raise RuntimeError(f"Failed to parse ADE20K class names from: {path}")
    return out


def _class_kind(name: str) -> str:
    n = name.lower()
    if any(k in n for k in OBJECT_KEYWORDS):
        return "OBJECT"
    if any(k in n for k in TEXTURE_KEYWORDS):
        return "TEXTURE_SURFACE"
    if any(k in n for k in AMBIGUOUS_KEYWORDS):
        return "AMBIGUOUS"
    return "OBJECT"


def build_ade20k_class_map(ade_root: Path) -> dict[int, str]:
    names = _parse_object_info(ade_root / "objectInfo150.txt")
    out = {0: "IGNORE"}
    for idx, name in names.items():
        out[int(idx)] = _class_kind(name)
    return out


def _decode_ade_label(mask_arr: np.ndarray) -> np.ndarray:
    if mask_arr.ndim == 2:
        return mask_arr.astype(np.int32)
    if mask_arr.ndim != 3:
        raise ValueError("Unsupported ADE mask shape")

    c0 = mask_arr[:, :, 0].astype(np.int32)
    c1 = mask_arr[:, :, 1].astype(np.int32)
    c2 = mask_arr[:, :, 2].astype(np.int32)

    if np.array_equal(c0, c1) and np.array_equal(c1, c2):
        return c0
    if int(c1.max()) <= 150 and int(c0.max()) <= 20 and int(c2.max()) <= 20:
        return c1
    if int(c0.max()) <= 150 and int(c1.max()) <= 20 and int(c2.max()) <= 20:
        return c0

    c3 = ((c0 // 10) * 256 + c1).astype(np.int32)
    c4 = ((c0 * 256 + c1) // 10).astype(np.int32)
    cands = [c0, c1, c3, c4]
    best = c0
    best_score = -1
    for cand in cands:
        in_range = ((cand >= 0) & (cand <= 150)).mean()
        uniq = int(np.unique(cand).size)
        score = float(in_range) + 0.01 * min(uniq, 200)
        if score > best_score:
            best_score = score
            best = cand
    return best


def _entropy(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    p = r / max(1e-9, float(r.sum()))
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _pair_count_metrics(region_map: np.ndarray, large_ids: set[int], strong_min: int) -> tuple[int, float]:
    if not large_ids:
        return 0, 0.0
    large_mask = np.isin(region_map, np.fromiter(large_ids, dtype=np.int32))
    rm = np.where(large_mask, region_map, 0).astype(np.int32)

    def collect(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = (a != b) & (a > 0) & (b > 0)
        if not m.any():
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
        x = a[m].astype(np.int64)
        y = b[m].astype(np.int64)
        lo = np.minimum(x, y)
        hi = np.maximum(x, y)
        return lo, hi

    lo_h, hi_h = collect(rm[:, :-1], rm[:, 1:])
    lo_v, hi_v = collect(rm[:-1, :], rm[1:, :])

    if lo_h.size + lo_v.size == 0:
        return 0, 0.0

    lo = np.concatenate([lo_h, lo_v], axis=0)
    hi = np.concatenate([hi_h, hi_v], axis=0)
    max_id = int(max(1, int(hi.max()) + 1))
    code = lo * max_id + hi
    _, cnt = np.unique(code, return_counts=True)
    strong = int((cnt >= int(strong_min)).sum())
    boundary_total = int(cnt.sum())
    return strong, float(boundary_total)


def _analyze_one(
    row: dict[str, Any],
    class_map: dict[int, str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    image_id = str(row["image_id"])
    image_path = Path(str(row["image_path"]))
    ann_path = Path(str(row["annotation_ref"]))

    try:
        with Image.open(ann_path) as im:
            mask_arr = np.asarray(im)
        labels = _decode_ade_label(mask_arr)
        h, w = int(labels.shape[0]), int(labels.shape[1])
        total = float(max(1, h * w))

        uniq, counts = np.unique(labels, return_counts=True)
        class_counts = {int(c): int(n) for c, n in zip(uniq.tolist(), counts.tolist())}
        texture_ids = [cid for cid, _ in class_counts.items() if class_map.get(cid, "OBJECT") == "TEXTURE_SURFACE"]
        object_pixels = float(sum(v for cid, v in class_counts.items() if class_map.get(cid, "OBJECT") == "OBJECT"))
        texture_pixels = float(sum(v for cid, v in class_counts.items() if class_map.get(cid, "OBJECT") == "TEXTURE_SURFACE"))
        ambiguous_pixels = float(sum(v for cid, v in class_counts.items() if class_map.get(cid, "OBJECT") == "AMBIGUOUS"))

        object_fraction = object_pixels / total
        texture_fraction = texture_pixels / total
        ambiguous_fraction = ambiguous_pixels / total

        # Stage A proxy stats from per-class areas (cheap and stable for ADE masks).
        area_ratios = np.asarray([float(v) / total for cid, v in class_counts.items() if int(cid) != 0], dtype=np.float64)
        n_masks = int(area_ratios.size)
        largest_ratio = float(area_ratios.max()) if area_ratios.size else 0.0
        median_ratio = float(np.median(area_ratios)) if area_ratios.size else 0.0
        small_threshold = float(cfg.get("small_threshold", 0.001))
        small_frac = float((area_ratios < small_threshold).mean()) if area_ratios.size else 0.0
        area_entropy = _entropy(area_ratios)
        stage_a_score = _stage_a_rwtd_score(
            n_masks=n_masks,
            largest_ratio=largest_ratio,
            median_ratio=median_ratio,
            small_frac=small_frac,
            entropy=area_entropy,
        )

        # Texture connected components and adjacency metrics.
        region_map = np.zeros((h, w), dtype=np.int32)
        texture_areas: list[int] = []
        offset = 0
        for cid in texture_ids:
            cls_mask = labels == int(cid)
            if not cls_mask.any():
                continue
            cc, ncc = ndimage.label(cls_mask.astype(np.uint8), structure=_STRUCT)
            ncc_i = int(ncc)
            if ncc_i <= 0:
                continue
            m = cc > 0
            region_map[m] = cc[m] + offset
            comp_areas = np.bincount(cc.ravel())[1:]
            texture_areas.extend(int(x) for x in comp_areas.tolist() if int(x) > 0)
            offset += ncc_i

        min_region_frac = float(cfg.get("min_large_region_frac", 0.08))
        min_area = min_region_frac * total
        large_ids = {i + 1 for i, a in enumerate(texture_areas) if float(a) >= min_area}
        strong_min = int(cfg.get("strong_boundary_min_pixels", 40))
        strong_count, boundary_total = _pair_count_metrics(region_map, large_ids, strong_min=strong_min)
        perimeter = float(max(1, 2 * (h + w)))
        boundary_norm = boundary_total / perimeter
        large_areas = [int(texture_areas[i - 1]) for i in sorted(large_ids) if 0 <= i - 1 < len(texture_areas)]
        balance = _balance_score(large_areas)
        geom_score = _compute_texture_boundary_score(
            large_count=len(large_ids),
            strong_count=strong_count,
            boundary_norm=boundary_norm,
            balance=balance,
            object_fraction=object_fraction,
        )

        status = "ok"
        err = None
    except Exception as exc:  # pragma: no cover
        h, w = int(row.get("height", 0) or 0), int(row.get("width", 0) or 0)
        if h <= 0 or w <= 0:
            h, w = 0, 0
        n_masks = 0
        largest_ratio = 0.0
        median_ratio = 0.0
        small_frac = 0.0
        area_entropy = 0.0
        stage_a_score = 0.0
        object_fraction = 1.0
        texture_fraction = 0.0
        ambiguous_fraction = 0.0
        geom_score = 0.0
        strong_count = 0
        boundary_norm = 0.0
        large_ids = set()
        status = "unknown"
        err = str(exc)

    return {
        "image_id": image_id,
        "image_path": str(image_path.resolve()),
        "annotation_ref": str(ann_path.resolve()),
        "dataset": "ade20k",
        "width": int(w),
        "height": int(h),
        "file_size_bytes": int(image_path.stat().st_size) if image_path.exists() else 0,
        "resolution_ok": bool(min(w, h) >= int(cfg.get("min_short_side", 256))),
        "batch_id": 0,
        "stageA_status": status,
        "stageA_n_masks": int(n_masks),
        "stageA_largest_ratio": float(largest_ratio),
        "stageA_median_ratio": float(median_ratio),
        "stageA_small_frac": float(small_frac),
        "stageA_area_entropy": float(area_entropy),
        "stageA_rwtd_score": float(stage_a_score),
        "stageA_pass": bool(len(large_ids) >= 2 and geom_score >= 55.0),
        "stageA_error": err,
        "geom_status": status,
        "geom_texture_boundary_score": float(geom_score),
        "geom_object_fraction": float(object_fraction),
        "geom_texture_fraction": float(texture_fraction),
        "geom_ambiguous_fraction": float(ambiguous_fraction),
        "geom_num_large_texture_regions": int(len(large_ids)),
        "geom_num_strong_boundaries": int(strong_count),
        "geom_boundary_norm": float(boundary_norm),
        "geom_mask_rel": None,
        "geom_overlay_rel": None,
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
        "labels_present_raw": "|".join(str(x) for x in sorted([int(k) for k in class_counts.keys()])) if status == "ok" else "",
    }


def discover_ade20k_pairs(ade_root: Path) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for split in ("training", "validation"):
        img_dir = ade_root / "images" / split
        ann_dir = ade_root / "annotations" / split
        if not img_dir.exists() or not ann_dir.exists():
            continue
        anns = sorted(ann_dir.glob("*.png"))
        for ann_path in anns:
            stem = ann_path.stem
            img_path = img_dir / f"{stem}.jpg"
            if not img_path.exists():
                alt = list(img_dir.glob(f"{stem}.*"))
                if not alt:
                    continue
                img_path = alt[0]
            pairs.append(
                {
                    "image_id": f"{split}_{stem}",
                    "image_path": str(img_path.resolve()),
                    "annotation_ref": str(ann_path.resolve()),
                    "split": split,
                }
            )
    return pairs


def run_ade20k_full_eval(
    *,
    ade_root: Path,
    out_root: Path,
    cfg: dict[str, Any],
    selected_min: float,
    borderline_min: float,
    workers: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    class_map = build_ade20k_class_map(ade_root)
    write_json(out_root / "ade20k_class_map_summary.json", {
        "n_classes": int(len(class_map)),
        "counts": {
            "TEXTURE_SURFACE": int(sum(1 for v in class_map.values() if v == "TEXTURE_SURFACE")),
            "OBJECT": int(sum(1 for v in class_map.values() if v == "OBJECT")),
            "AMBIGUOUS": int(sum(1 for v in class_map.values() if v == "AMBIGUOUS")),
            "IGNORE": int(sum(1 for v in class_map.values() if v == "IGNORE")),
        },
    })

    pairs = discover_ade20k_pairs(ade_root)
    if not pairs:
        raise RuntimeError(f"No ADE20K pairs found under: {ade_root}")

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [ex.submit(_analyze_one, row, class_map, cfg) for row in pairs]
        for f in tqdm(as_completed(futs), total=len(futs), desc="ade20k_eval", unit="img"):
            rows.append(f.result())

    df = pd.DataFrame(rows)
    df = apply_texture_priority_scoring(df, selected_min=float(selected_min), borderline_min=float(borderline_min))
    df = df.sort_values("review_score", ascending=False).reset_index(drop=True)

    total = int(len(df))
    selected = int(df["final_selected"].fillna(False).sum())
    borderline = int(df["final_borderline"].fillna(False).sum())
    rejected = total - selected - borderline
    summary = {
        "n_total": total,
        "selected_min": float(selected_min),
        "borderline_min": float(borderline_min),
        "selected_count": selected,
        "borderline_count": borderline,
        "rejected_count": rejected,
        "selected_ratio": float(selected / max(1, total)),
        "score_quantiles": {
            "q10": float(df["review_score"].quantile(0.10)),
            "q25": float(df["review_score"].quantile(0.25)),
            "q50": float(df["review_score"].quantile(0.50)),
            "q75": float(df["review_score"].quantile(0.75)),
            "q90": float(df["review_score"].quantile(0.90)),
            "max": float(df["review_score"].max()),
        },
    }
    write_json(out_root / "ade20k_eval_summary.json", summary)
    return df, summary


def render_ade20k_review_assets(
    *,
    df: pd.DataFrame,
    batch_dir: Path,
    class_map: dict[int, str],
    cfg: dict[str, Any],
    max_items: int | None = None,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    if max_items is not None and max_items > 0 and len(out) > int(max_items):
        out = out.head(int(max_items)).copy()

    masks_dir = ensure_dir(batch_dir / "review" / "masks")
    overlays_dir = ensure_dir(batch_dir / "review" / "overlays")

    object_ids = np.asarray([k for k, v in class_map.items() if v == "OBJECT"], dtype=np.int32)
    texture_ids = np.asarray([k for k, v in class_map.items() if v == "TEXTURE_SURFACE"], dtype=np.int32)
    ambiguous_ids = np.asarray([k for k, v in class_map.items() if v == "AMBIGUOUS"], dtype=np.int32)
    min_region_frac = float(cfg.get("min_large_region_frac", 0.08))

    out["geom_mask_rel"] = None
    out["geom_overlay_rel"] = None

    for i, row in tqdm(out.iterrows(), total=len(out), desc="ade20k_render", unit="img"):
        image_path = Path(str(row.get("image_path", "")))
        ann_path = Path(str(row.get("annotation_ref", "")))
        image_id = str(row.get("image_id", ""))
        if not image_id or (not image_path.exists()) or (not ann_path.exists()):
            continue
        try:
            with Image.open(ann_path) as im:
                labels = _decode_ade_label(np.asarray(im))

            h, w = int(labels.shape[0]), int(labels.shape[1])
            total = float(max(1, h * w))
            raw_mask = np.zeros((h, w), dtype=np.uint8)

            if texture_ids.size > 0:
                raw_mask[np.isin(labels, texture_ids)] = 1
            if ambiguous_ids.size > 0:
                raw_mask[np.isin(labels, ambiguous_ids)] = 3
            if object_ids.size > 0:
                raw_mask[np.isin(labels, object_ids)] = 2

            region_map = np.zeros((h, w), dtype=np.int32)
            areas: list[int] = []
            offset = 0
            for cid in texture_ids.tolist():
                cls_mask = labels == int(cid)
                if not cls_mask.any():
                    continue
                cc, ncc = ndimage.label(cls_mask.astype(np.uint8), structure=_STRUCT)
                ncc_i = int(ncc)
                if ncc_i <= 0:
                    continue
                m = cc > 0
                region_map[m] = cc[m] + offset
                comp_areas = np.bincount(cc.ravel())[1:]
                areas.extend(int(x) for x in comp_areas.tolist() if int(x) > 0)
                offset += ncc_i

            min_area = min_region_frac * total
            large_ids = {rid for rid, area in enumerate(areas, start=1) if float(area) >= min_area}
            boundary_pixels, _ = _boundary_pixels_and_counts(region_map, large_ids=large_ids)

            mask_out = masks_dir / f"{image_id}.jpg"
            overlay_out = overlays_dir / f"{image_id}.jpg"
            _render_assets(
                image_path=image_path,
                raw_mask=raw_mask,
                texture_region_map=region_map,
                boundary_pixels=boundary_pixels,
                mask_out=mask_out,
                overlay_out=overlay_out,
            )
            out.at[i, "geom_mask_rel"] = str(mask_out.resolve())
            out.at[i, "geom_overlay_rel"] = str(overlay_out.resolve())
        except Exception:
            continue

    return out
