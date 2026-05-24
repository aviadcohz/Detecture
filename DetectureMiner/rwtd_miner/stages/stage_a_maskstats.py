from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from rwtd_miner.utils.mat_annotations import load_mat_as_annotation_payload

try:
    from pycocotools import mask as mask_utils
except Exception:  # pragma: no cover
    mask_utils = None


# Per-process caches. Each worker process keeps its own copy.
_JSON_FILE_CACHE: dict[str, Any] = {}
_COCO_LOOKUP_CACHE: dict[int, tuple[dict[str, Any], dict[Any, tuple[int | None, int | None]], dict[Any, list[dict[str, Any]]]]] = {}


def _decode_uncompressed_rle_area(seg: dict) -> float | None:
    counts = seg.get("counts")
    if not isinstance(counts, list):
        return None
    area = 0
    val = 0
    for run in counts:
        run_i = int(run)
        if val == 1:
            area += run_i
        val = 1 - val
    return float(area)


def _segmentation_area(segmentation: Any) -> float | None:
    if segmentation is None:
        return None

    if isinstance(segmentation, np.ndarray):
        if segmentation.ndim >= 2:
            return float((segmentation > 0).sum())
        return None

    if isinstance(segmentation, dict):
        if mask_utils is not None:
            try:
                return float(mask_utils.area(segmentation))
            except Exception:
                pass
        return _decode_uncompressed_rle_area(segmentation)

    if isinstance(segmentation, list):
        # Polygon list [x1,y1,x2,y2,...] or list-of-polygons
        polys = segmentation
        if polys and isinstance(polys[0], (int, float)):
            polys = [polys]
        area_sum = 0.0
        for poly in polys:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            xs = np.asarray(poly[0::2], dtype=np.float32)
            ys = np.asarray(poly[1::2], dtype=np.float32)
            if xs.size < 3:
                continue
            area = 0.5 * float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))
            area_sum += area
        return area_sum if area_sum > 0 else None

    return None


def _entry_image_key(entry: dict) -> str | None:
    for key in ("image_id", "id", "sa_id"):
        value = entry.get(key)
        if value is not None:
            return str(value)
    image = entry.get("image")
    if isinstance(image, dict):
        if image.get("id") is not None:
            return str(image.get("id"))
        if image.get("image_id") is not None:
            return str(image.get("image_id"))
        if image.get("file_name"):
            return Path(str(image.get("file_name"))).stem
    for key in ("file_name", "image_path", "image"):
        value = entry.get(key)
        if value:
            return Path(str(value)).stem
    return None


def _extract_annotations_from_payload(payload: Any, image_id: str, image_path: str) -> tuple[list[dict], int | None, int | None]:
    image_stem = Path(image_path).stem

    if isinstance(payload, dict):
        if isinstance(payload.get("annotations"), list) and not isinstance(payload.get("images"), list):
            image_meta = payload.get("image") if isinstance(payload.get("image"), dict) else {}
            w = image_meta.get("width")
            h = image_meta.get("height")
            return payload.get("annotations", []), int(w) if w else None, int(h) if h else None

        if isinstance(payload.get("images"), list) and isinstance(payload.get("annotations"), list):
            key_to_lookup, dims_by_lookup, anns_by_lookup = _build_coco_lookup(payload)
            image_id_lookup = key_to_lookup.get(image_id) or key_to_lookup.get(image_stem)
            if image_id_lookup is None:
                return [], None, None
            width, height = dims_by_lookup.get(image_id_lookup, (None, None))
            selected = anns_by_lookup.get(image_id_lookup, [])

            if selected and all(isinstance(a, dict) and isinstance(a.get("segments_info"), list) for a in selected):
                flattened: list[dict] = []
                for ann in selected:
                    segs = ann.get("segments_info", [])
                    if not isinstance(segs, list):
                        continue
                    for seg in segs:
                        if not isinstance(seg, dict):
                            continue
                        flattened.append(
                            {
                                "area": seg.get("area", 0.0),
                                "bbox": seg.get("bbox", [0, 0, 0, 0]),
                                "category_id": seg.get("category_id"),
                                "segmentation": seg.get("segmentation"),
                            }
                        )
                selected = flattened

            return selected, width, height

        if isinstance(payload.get("data"), list):
            payload = payload.get("data")

    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            key = _entry_image_key(entry)
            if key not in {image_id, image_stem}:
                continue
            anns = entry.get("annotations") if isinstance(entry.get("annotations"), list) else entry.get("segments", [])
            image_meta = entry.get("image") if isinstance(entry.get("image"), dict) else {}
            w = image_meta.get("width") or entry.get("width")
            h = image_meta.get("height") or entry.get("height")
            return anns if isinstance(anns, list) else [], int(w) if w else None, int(h) if h else None

    return [], None, None


def _compute_entropy(area_ratios: np.ndarray) -> float:
    if area_ratios.size == 0:
        return 0.0
    p = area_ratios / max(1e-9, area_ratios.sum())
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _stage_a_rwtd_score(
    *,
    n_masks: int,
    largest_ratio: float,
    median_ratio: float,
    small_frac: float,
    entropy: float,
) -> float:
    n_score = min(1.0, n_masks / 120.0)
    largest_score = 1.0 - min(1.0, largest_ratio / 0.65)
    median_score = 1.0 - min(1.0, median_ratio / 0.03)
    small_score = min(1.0, small_frac / 0.55)
    entropy_score = min(1.0, entropy / 4.0)
    weighted = (
        0.28 * n_score
        + 0.24 * largest_score
        + 0.18 * median_score
        + 0.18 * small_score
        + 0.12 * entropy_score
    )
    return float(max(0.0, min(100.0, weighted * 100.0)))


def _annotation_payload_from_ref(annotation_ref: str) -> Any:
    def _read_json_cached(path: Path) -> Any:
        key = str(path)
        payload = _JSON_FILE_CACHE.get(key)
        if payload is None:
            payload = json.loads(path.read_text(encoding="utf-8"))
            _JSON_FILE_CACHE[key] = payload
        return payload

    if "::" in annotation_ref:
        path_str, idx_str = annotation_ref.rsplit("::", 1)
        path = Path(path_str)
        if path.suffix.lower() == ".mat":
            return load_mat_as_annotation_payload(path)
        payload = _read_json_cached(path)
        if idx_str.isdigit() and isinstance(payload, list):
            idx = int(idx_str)
            if 0 <= idx < len(payload):
                return payload[idx]
        return payload
    path = Path(annotation_ref)
    if path.suffix.lower() == ".mat":
        return load_mat_as_annotation_payload(path)
    return _read_json_cached(path)


def _build_coco_lookup(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[Any, tuple[int | None, int | None]], dict[Any, list[dict[str, Any]]]]:
    pid = id(payload)
    cached = _COCO_LOOKUP_CACHE.get(pid)
    if cached is not None:
        return cached

    key_to_lookup: dict[str, Any] = {}
    dims_by_lookup: dict[Any, tuple[int | None, int | None]] = {}
    anns_by_lookup: dict[Any, list[dict[str, Any]]] = {}

    images = payload.get("images", [])
    anns = payload.get("annotations", [])

    if isinstance(images, list):
        for im in images:
            if not isinstance(im, dict):
                continue
            lookup = im.get("id") if im.get("id") is not None else im.get("image_id")
            if lookup is None:
                continue
            width = int(im.get("width")) if im.get("width") else None
            height = int(im.get("height")) if im.get("height") else None
            dims_by_lookup[lookup] = (width, height)

            candidates: list[str] = [str(lookup)]
            if im.get("image_id") is not None:
                candidates.append(str(im.get("image_id")))
            file_name = im.get("file_name")
            if file_name:
                candidates.append(Path(str(file_name)).stem)
            for cand in candidates:
                if cand and cand not in key_to_lookup:
                    key_to_lookup[cand] = lookup

    if isinstance(anns, list):
        for ann in anns:
            if not isinstance(ann, dict):
                continue
            lookup = ann.get("image_id")
            if lookup is None:
                continue
            anns_by_lookup.setdefault(lookup, []).append(ann)

    cached = (key_to_lookup, dims_by_lookup, anns_by_lookup)
    _COCO_LOOKUP_CACHE[pid] = cached
    return cached


def _worker(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    image_area = None
    width = row.get("width")
    height = row.get("height")
    if width and height:
        image_area = float(width) * float(height)

    annotation_ref = row.get("annotation_ref")
    if not annotation_ref:
        return {
            "image_id": row["image_id"],
            "stageA_status": "unknown",
            "stageA_n_masks": None,
            "stageA_largest_ratio": None,
            "stageA_median_ratio": None,
            "stageA_small_frac": None,
            "stageA_area_entropy": None,
            "stageA_rwtd_score": None,
            "stageA_pass": None,
            "stageA_error": "missing_annotation",
        }

    try:
        payload = _annotation_payload_from_ref(str(annotation_ref))
        anns, ann_w, ann_h = _extract_annotations_from_payload(payload, row["image_id"], row["image_path"])
        if image_area is None and ann_w and ann_h:
            image_area = float(ann_w * ann_h)
        if image_area is None:
            return {
                "image_id": row["image_id"],
                "stageA_status": "unknown",
                "stageA_n_masks": None,
                "stageA_largest_ratio": None,
                "stageA_median_ratio": None,
                "stageA_small_frac": None,
                "stageA_area_entropy": None,
                "stageA_rwtd_score": None,
                "stageA_pass": None,
                "stageA_error": "missing_image_area",
            }

        areas: list[float] = []
        for ann in anns:
            if not isinstance(ann, dict):
                continue
            area = ann.get("area")
            if area is None:
                area = _segmentation_area(ann.get("segmentation"))
            if area is None:
                continue
            area_f = float(area)
            if area_f > 0:
                areas.append(area_f)

        if not areas:
            return {
                "image_id": row["image_id"],
                "stageA_status": "unknown",
                "stageA_n_masks": 0,
                "stageA_largest_ratio": 0.0,
                "stageA_median_ratio": 0.0,
                "stageA_small_frac": 0.0,
                "stageA_area_entropy": 0.0,
                "stageA_rwtd_score": 0.0,
                "stageA_pass": False,
                "stageA_error": "no_mask_areas",
            }

        ratios = np.asarray(areas, dtype=np.float64) / max(1.0, image_area)
        n_masks = int(ratios.size)
        largest = float(ratios.max())
        median = float(np.median(ratios))
        small_frac = float((ratios < float(cfg.get("small_threshold", 0.001))).mean())
        entropy = _compute_entropy(ratios)

        pass_flag = (
            n_masks >= int(cfg.get("n_masks_min", 80))
            and largest <= float(cfg.get("largest_mask_ratio_max", 0.60))
            and median <= float(cfg.get("median_mask_ratio_max", 0.02))
            and small_frac >= float(cfg.get("small_mask_fraction_min", 0.30))
        )
        if bool(cfg.get("use_entropy", True)):
            pass_flag = pass_flag and (entropy >= float(cfg.get("entropy_min", 2.0)))

        return {
            "image_id": row["image_id"],
            "stageA_status": "ok",
            "stageA_n_masks": n_masks,
            "stageA_largest_ratio": largest,
            "stageA_median_ratio": median,
            "stageA_small_frac": small_frac,
            "stageA_area_entropy": entropy,
            "stageA_rwtd_score": _stage_a_rwtd_score(
                n_masks=n_masks,
                largest_ratio=largest,
                median_ratio=median,
                small_frac=small_frac,
                entropy=entropy,
            ),
            "stageA_pass": bool(pass_flag),
            "stageA_error": None,
        }
    except Exception as exc:
        return {
            "image_id": row["image_id"],
            "stageA_status": "unknown",
            "stageA_n_masks": None,
            "stageA_largest_ratio": None,
            "stageA_median_ratio": None,
            "stageA_small_frac": None,
            "stageA_area_entropy": None,
            "stageA_rwtd_score": None,
            "stageA_pass": None,
            "stageA_error": str(exc),
        }


def run_stage_a(batch_df: pd.DataFrame, cfg: dict[str, Any], workers: int) -> pd.DataFrame:
    rows = batch_df[["image_id", "image_path", "annotation_ref", "width", "height"]].to_dict("records")
    results: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        mapped = ex.map(_worker, rows, repeat(cfg), chunksize=64)
        for out_row in tqdm(mapped, total=len(rows), desc="stageA_maskstats", unit="img"):
            results.append(out_row)

    out = pd.DataFrame(results)
    return batch_df.merge(out, on="image_id", how="left")


def apply_stage_a_fallbacks(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    allow = bool(cfg.get("allow_clip_fallback_if_unknown", True))
    if allow:
        unknown = df["stageA_status"].fillna("unknown") == "unknown"
        df.loc[unknown, "stageA_pass"] = True
    df["stageA_pass"] = df["stageA_pass"].fillna(False).astype(bool)
    return df


def stage_a_pass_rate(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    return float(df["stageA_pass"].fillna(False).mean())


def stage_a_strictness_warning(pass_rate: float) -> str | None:
    if pass_rate < 0.001:
        return "Stage A appears too strict (<0.1% pass)."
    if pass_rate > 0.5:
        return "Stage A appears too loose (>50% pass)."
    return None
