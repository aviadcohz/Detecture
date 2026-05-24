from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from rwtd_miner.utils.io import ensure_dir
from rwtd_miner.utils.mat_annotations import load_mat_as_annotation_payload

try:
    from pycocotools import mask as mask_utils
except Exception:  # pragma: no cover
    mask_utils = None


# Caches make shard-style COCO annotation refs practical:
# many rows point to the same large JSON file.
_JSON_FILE_CACHE: dict[str, Any] = {}
_COCO_LOOKUP_CACHE: dict[int, tuple[dict[str, Any], dict[Any, tuple[int | None, int | None]], dict[Any, list[dict[str, Any]]]]] = {}


def _color_for_id(idx: int) -> tuple[int, int, int]:
    seed = (1103515245 * (idx + 12345) + 12345) & 0x7FFFFFFF
    r = 48 + (seed % 192)
    g = 48 + ((seed // 193) % 192)
    b = 48 + ((seed // (193 * 193)) % 192)
    return int(r), int(g), int(b)


def _read_annotation_payload(annotation_ref: str) -> dict[str, Any] | None:
    if not annotation_ref:
        return None

    def _read_json_cached(path: Path) -> Any:
        key = str(path)
        payload = _JSON_FILE_CACHE.get(key)
        if payload is None:
            payload = json.loads(path.read_text(encoding="utf-8"))
            _JSON_FILE_CACHE[key] = payload
        return payload

    if "::" in annotation_ref:
        p, idx = annotation_ref.rsplit("::", 1)
        path = Path(p)
        if path.suffix.lower() == ".mat":
            payload = load_mat_as_annotation_payload(path)
            return payload if isinstance(payload, dict) else None
        payload = _read_json_cached(path)
        if idx.isdigit() and isinstance(payload, list):
            i = int(idx)
            if 0 <= i < len(payload) and isinstance(payload[i], dict):
                return payload[i]
        if isinstance(payload, dict):
            return payload
        return None
    path = Path(annotation_ref)
    if path.suffix.lower() == ".mat":
        payload = load_mat_as_annotation_payload(path)
        return payload if isinstance(payload, dict) else None
    payload = _read_json_cached(path)
    return payload if isinstance(payload, dict) else None


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


def _extract_annotations_for_image(
    payload: dict[str, Any],
    *,
    image_id: str,
    image_path: Path,
) -> tuple[list[dict[str, Any]], int | None, int | None]:
    # Per-image JSON payload (SA-1B-like).
    if isinstance(payload.get("annotations"), list) and not isinstance(payload.get("images"), list):
        image_meta = payload.get("image") if isinstance(payload.get("image"), dict) else {}
        w = image_meta.get("width")
        h = image_meta.get("height")
        return payload.get("annotations", []), int(w) if w else None, int(h) if h else None

    # COCO-style shard payload.
    if isinstance(payload.get("images"), list) and isinstance(payload.get("annotations"), list):
        key_to_lookup, dims_by_lookup, anns_by_lookup = _build_coco_lookup(payload)
        image_stem = image_path.stem
        image_lookup = key_to_lookup.get(image_id) or key_to_lookup.get(image_stem)
        if image_lookup is None:
            return [], None, None
        width, height = dims_by_lookup.get(image_lookup, (None, None))
        selected = anns_by_lookup.get(image_lookup, [])

        # Panoptic JSON stores segments under one annotation row per image.
        # Flatten into pseudo-annotations so downstream geometry/scoring can reuse
        # one code path. Segmentation masks may be missing if panoptic PNGs are not
        # available; in that case decode just skips those entries.
        if selected and all(isinstance(a, dict) and isinstance(a.get("segments_info"), list) for a in selected):
            flattened: list[dict[str, Any]] = []
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
                            "segmentation": seg.get("segmentation", {}),
                        }
                    )
            selected = flattened

        return selected, width, height

    return [], None, None


def _segment_stats(ann: dict, width: int, height: int) -> dict[str, float | bool]:
    area = float(ann.get("area", 0.0))
    bbox = ann.get("bbox", [0, 0, 0, 0])
    if not isinstance(bbox, list) or len(bbox) < 4:
        bbox = [0, 0, 0, 0]
    x, y, bw, bh = [float(v) for v in bbox[:4]]
    fill = area / max(1.0, bw * bh)
    aspect = bw / max(1.0, bh)
    iou = float(ann.get("predicted_iou", 0.0))
    stability = float(ann.get("stability_score", 0.0))
    area_frac = area / max(1.0, float(width * height))
    border_touch = (x <= 2.0) or (y <= 2.0) or ((x + bw) >= (width - 2.0)) or ((y + bh) >= (height - 2.0))
    cx = x + 0.5 * bw
    cy = y + 0.5 * bh
    nx = (cx / max(1.0, float(width))) - 0.5
    ny = (cy / max(1.0, float(height))) - 0.5
    center_distance = min(1.0, float(np.sqrt(nx * nx + ny * ny) / 0.70710678))
    return {
        "area_frac": area_frac,
        "bbox_fill_ratio": fill,
        "aspect_ratio": aspect,
        "predicted_iou": iou,
        "stability_score": stability,
        "center_distance": center_distance,
        "border_touch": border_touch,
    }


def _classify_segment(st: dict[str, float | bool]) -> int:
    area_frac = float(st["area_frac"])
    fill = float(st["bbox_fill_ratio"])
    aspect = float(st["aspect_ratio"])
    iou = float(st["predicted_iou"])
    stability = float(st["stability_score"])
    center_distance = float(st["center_distance"])
    border_touch = bool(st["border_touch"])

    if area_frac < 0.004:
        return 2
    if fill < 0.18:
        return 2
    if (aspect > 6.0 or aspect < 0.16) and ((not border_touch) or area_frac <= 0.012):
        return 2
    if (not border_touch) and area_frac < 0.22 and fill > 0.55 and center_distance < 0.68 and iou > 0.88:
        return 2
    if area_frac > 0.18 and fill > 0.35:
        return 1
    if area_frac > 0.06 and iou > 0.86 and stability > 0.9 and fill > 0.35:
        return 1
    if border_touch and area_frac > 0.05:
        return 1
    return 3


def _decode_segmentation(segmentation: Any, *, width: int, height: int) -> np.ndarray | None:
    if isinstance(segmentation, np.ndarray):
        m = segmentation
        if m.ndim == 3:
            m = m[:, :, 0]
        if m.ndim != 2:
            return None
        return (m > 0).astype(np.uint8)
    if mask_utils is None:
        return None
    if isinstance(segmentation, dict):
        try:
            m = mask_utils.decode(segmentation)
            if m.ndim == 3:
                m = m[:, :, 0]
            return (m > 0).astype(np.uint8)
        except Exception:
            return None
    if isinstance(segmentation, list):
        try:
            if segmentation and isinstance(segmentation[0], (int, float)):
                rles = mask_utils.frPyObjects([segmentation], height, width)
            else:
                rles = mask_utils.frPyObjects(segmentation, height, width)
            m = mask_utils.decode(rles)
            if m.ndim == 3:
                m = np.any(m > 0, axis=2)
            return (m > 0).astype(np.uint8)
        except Exception:
            return None
    return None


def _boundary_pair_pixels(region_map: np.ndarray, large_ids: set[int]) -> dict[tuple[int, int], set[tuple[int, int]]]:
    pair_pixels: dict[tuple[int, int], set[tuple[int, int]]] = {}
    if not large_ids:
        return pair_pixels

    # Horizontal neighbors.
    a = region_map[:, :-1]
    b = region_map[:, 1:]
    yx = np.where((a != b) & (a > 0) & (b > 0))
    for y, x in zip(yx[0].tolist(), yx[1].tolist()):
        ra = int(a[y, x])
        rb = int(b[y, x])
        if ra in large_ids and rb in large_ids and ra != rb:
            k = (ra, rb) if ra < rb else (rb, ra)
            pp = pair_pixels.setdefault(k, set())
            pp.add((int(y), int(x)))
            pp.add((int(y), int(x + 1)))

    # Vertical neighbors.
    a = region_map[:-1, :]
    b = region_map[1:, :]
    yx = np.where((a != b) & (a > 0) & (b > 0))
    for y, x in zip(yx[0].tolist(), yx[1].tolist()):
        ra = int(a[y, x])
        rb = int(b[y, x])
        if ra in large_ids and rb in large_ids and ra != rb:
            k = (ra, rb) if ra < rb else (rb, ra)
            pp = pair_pixels.setdefault(k, set())
            pp.add((int(y), int(x)))
            pp.add((int(y + 1), int(x)))

    return pair_pixels


def _boundary_pixels_and_counts(region_map: np.ndarray, large_ids: set[int]) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]]:
    pair_pixels = _boundary_pair_pixels(region_map, large_ids=large_ids)
    boundary_pixels: set[tuple[int, int]] = set()
    pair_counts: dict[tuple[int, int], int] = {}
    for k, pix in pair_pixels.items():
        pair_counts[k] = int(len(pix))
        boundary_pixels.update(pix)
    return boundary_pixels, pair_counts


def _build_coarse_dense_label_maps(
    decoded_masks: list[np.ndarray],
    *,
    out_h: int,
    out_w: int,
    total: float,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    min_frac = float(cfg.get("dense_label_coarse_region_min_frac", 0.04))
    max_regions = int(cfg.get("dense_label_coarse_region_max_regions", 6))
    fallback_min_frac = float(cfg.get("dense_label_fallback_region_min_frac", 0.02))

    areas_masks: list[tuple[int, np.ndarray]] = []
    for bm in decoded_masks:
        area = int((bm > 0).sum())
        if area <= 0:
            continue
        areas_masks.append((area, (bm > 0).astype(np.uint8)))
    areas_masks.sort(key=lambda x: x[0], reverse=True)

    def _pick(thr_frac: float) -> list[tuple[int, np.ndarray]]:
        thr = float(thr_frac) * float(total)
        picked = [x for x in areas_masks if float(x[0]) >= thr]
        return picked[:max_regions] if max_regions > 0 else picked

    picked = _pick(min_frac)
    if len(picked) < 2:
        picked = _pick(fallback_min_frac)

    raw_mask = np.zeros((out_h, out_w), dtype=np.uint8)
    texture_region_map = np.zeros((out_h, out_w), dtype=np.int32)
    for rid, (_, bm) in enumerate(picked, start=1):
        write = (texture_region_map == 0) & (bm == 1)
        if not write.any():
            continue
        raw_mask[write] = np.uint8(1)
        texture_region_map[write] = int(rid)

    # Remaining masks are present but omitted as over-fine detail.
    if areas_masks:
        rest = np.zeros((out_h, out_w), dtype=np.uint8)
        for _, bm in areas_masks:
            rest = np.maximum(rest, bm)
        raw_mask[(raw_mask == 0) & (rest == 1)] = np.uint8(3)

    return raw_mask, texture_region_map


def _balance_score(areas: list[int]) -> float:
    if len(areas) < 2:
        return 0.0
    total = float(sum(areas))
    if total <= 0:
        return 0.0
    largest_ratio = max(areas) / total
    # best around 2-4 balanced regions.
    return float(max(0.0, min(1.0, 1.0 - largest_ratio)))


def _compute_texture_boundary_score(
    *,
    large_count: int,
    strong_count: int,
    boundary_norm: float,
    balance: float,
    object_fraction: float,
) -> float:
    region_term = min(1.0, large_count / 3.0)
    strong_term = min(1.0, strong_count / 2.0)
    boundary_term = min(1.0, boundary_norm / 0.35)
    raw = 100.0 * (0.28 * region_term + 0.34 * strong_term + 0.26 * boundary_term + 0.12 * balance)
    penalized = raw * (1.0 - min(0.75, object_fraction * 1.15))
    return float(max(0.0, min(100.0, penalized)))


def _render_assets(
    image_path: Path,
    raw_mask: np.ndarray,
    texture_region_map: np.ndarray,
    boundary_pixels: set[tuple[int, int]],
    mask_out: Path,
    overlay_out: Path,
) -> None:
    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"))

    h, w = raw_mask.shape
    if rgb.shape[0] != h or rgb.shape[1] != w:
        im = Image.fromarray(rgb)
        im = im.resize((w, h), Image.Resampling.BICUBIC)
        rgb = np.asarray(im)

    vis = np.zeros((h, w, 3), dtype=np.uint8)
    # Object and ambiguous colors.
    vis[raw_mask == 2] = np.array([255, 210, 0], dtype=np.uint8)
    vis[raw_mask == 3] = np.array([90, 225, 120], dtype=np.uint8)

    tex_ids = np.unique(texture_region_map[texture_region_map > 0]).tolist()
    for rid in tex_ids:
        vis[texture_region_map == int(rid)] = np.array(_color_for_id(int(rid)), dtype=np.uint8)

    overlay = (0.62 * rgb.astype(np.float32) + 0.38 * vis.astype(np.float32)).clip(0, 255).astype(np.uint8)
    for y, x in boundary_pixels:
        if 0 <= y < h and 0 <= x < w:
            overlay[y, x] = np.array([255, 40, 30], dtype=np.uint8)

    ensure_dir(mask_out.parent)
    ensure_dir(overlay_out.parent)
    Image.fromarray(vis).save(mask_out, quality=90)
    Image.fromarray(overlay).save(overlay_out, quality=90)


def _analyze_one(
    image_id: str,
    image_path: Path,
    annotation_ref: str,
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

    payload = _read_annotation_payload(annotation_ref)
    if not payload:
        return {
            "image_id": image_id,
            "geom_status": "missing_annotation",
            "geom_texture_boundary_score": None,
            "geom_object_fraction": None,
            "geom_texture_fraction": None,
            "geom_ambiguous_fraction": None,
            "geom_num_large_texture_regions": None,
            "geom_num_strong_boundaries": None,
            "geom_boundary_norm": None,
            "geom_mask_rel": None,
            "geom_overlay_rel": None,
        }

    anns, ann_w, ann_h = _extract_annotations_for_image(payload, image_id=image_id, image_path=image_path)
    if not anns:
        return {
            "image_id": image_id,
            "geom_status": "no_annotations",
            "geom_texture_boundary_score": None,
            "geom_object_fraction": None,
            "geom_texture_fraction": None,
            "geom_ambiguous_fraction": None,
            "geom_num_large_texture_regions": None,
            "geom_num_strong_boundaries": None,
            "geom_boundary_norm": None,
            "geom_mask_rel": None,
            "geom_overlay_rel": None,
        }

    w = int(ann_w or 0)
    h = int(ann_h or 0)
    if w <= 0 or h <= 0:
        with Image.open(image_path) as im:
            w, h = im.size

    target_long = int(cfg.get("mask_target_long_side", 960))
    if target_long > 0 and max(h, w) > target_long:
        scale = target_long / float(max(h, w))
        out_w = max(1, int(round(w * scale)))
        out_h = max(1, int(round(h * scale)))
    else:
        out_w, out_h = w, h

    max_segments = int(cfg.get("max_segments_decode_per_image", 80))
    anns_sorted = sorted(anns, key=lambda a: float(a.get("area", 0.0)), reverse=True)[:max_segments]

    total = float(out_h * out_w)
    dense_label_mode = bool(cfg.get("dense_label_use_coarse_boundaries_only", True)) and (
        str(payload.get("__rwtd_source", "")).strip() == "dense_label_map"
    )

    decoded_items: list[tuple[int, float, np.ndarray]] = []
    dense_masks: list[np.ndarray] = []
    for ann in anns_sorted:
        seg = ann.get("segmentation", {})
        bm = _decode_segmentation(seg, width=w, height=h)
        if bm is None:
            continue
        if bm.shape != (out_h, out_w):
            pil = Image.fromarray((bm * 255).astype(np.uint8), mode="L")
            pil = pil.resize((out_w, out_h), Image.Resampling.NEAREST)
            bm = (np.asarray(pil) > 0).astype(np.uint8)
        if dense_label_mode:
            dense_masks.append(bm)
            continue
        st = _segment_stats(ann, width=w, height=h)
        cls_id = _classify_segment(st)
        decoded_items.append((int(cls_id), float(ann.get("area", 0.0)), bm))

    if dense_label_mode:
        raw_mask, texture_region_map = _build_coarse_dense_label_maps(
            dense_masks,
            out_h=out_h,
            out_w=out_w,
            total=total,
            cfg=cfg,
        )
    else:
        raw_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        texture_region_map = np.zeros((out_h, out_w), dtype=np.int32)
        next_tex_id = 1

        for cls_id in [2, 3, 1]:
            cls_items = [x for x in decoded_items if x[0] == cls_id]
            if cls_id == 2:
                cls_items.sort(key=lambda x: x[1])
            else:
                cls_items.sort(key=lambda x: x[1], reverse=True)
            for _, _, bm in cls_items:
                write = (raw_mask == 0) & (bm == 1)
                if not write.any():
                    continue
                raw_mask[write] = np.uint8(cls_id)
                if cls_id == 1:
                    texture_region_map[write] = int(next_tex_id)
                    next_tex_id += 1

    obj_frac = float((raw_mask == 2).sum()) / max(1.0, total)
    tex_frac = float((raw_mask == 1).sum()) / max(1.0, total)
    amb_frac = float((raw_mask == 3).sum()) / max(1.0, total)

    # Use SAM texture instance map directly for boundaries between texture segments.
    tex_map = texture_region_map.astype(np.int32)
    max_tid = int(tex_map.max())
    areas = np.bincount(tex_map.ravel(), minlength=max_tid + 1)[1:] if max_tid > 0 else np.asarray([], dtype=np.int64)
    min_region_frac = float(cfg.get("min_large_region_frac", 0.08))
    if dense_label_mode:
        min_region_frac = float(cfg.get("dense_label_min_large_region_frac", min_region_frac))
    min_area = min_region_frac * total
    large_ids = {int(i + 1) for i, a in enumerate(areas.tolist()) if float(a) >= min_area}

    pair_pixels = _boundary_pair_pixels(tex_map, large_ids=large_ids)
    pair_counts = {k: len(v) for k, v in pair_pixels.items()}
    perimeter = float(2 * (out_h + out_w))
    min_pair_frac = float(cfg.get("boundary_pair_min_perimeter_frac", 0.012))
    if dense_label_mode:
        min_pair_frac = float(cfg.get("dense_label_boundary_pair_min_perimeter_frac", max(min_pair_frac, 0.03)))
    min_pair_pixels = max(1, int(round(min_pair_frac * perimeter)))
    pair_counts = {k: v for k, v in pair_counts.items() if int(v) >= min_pair_pixels}
    boundary_pixels: set[tuple[int, int]] = set()
    for k in pair_counts.keys():
        boundary_pixels.update(pair_pixels.get(k, set()))
    boundary_norm = float(sum(pair_counts.values())) / max(1.0, perimeter)

    strong_boundary_min = int(cfg.get("strong_boundary_min_pixels", 40))
    strong_count = int(sum(1 for v in pair_counts.values() if int(v) >= strong_boundary_min))
    large_areas = [int(areas[i - 1]) for i in sorted(large_ids) if 0 <= i - 1 < len(areas)]
    balance = _balance_score(large_areas)

    texture_boundary_score = _compute_texture_boundary_score(
        large_count=len(large_ids),
        strong_count=strong_count,
        boundary_norm=boundary_norm,
        balance=balance,
        object_fraction=obj_frac,
    )

    _render_assets(
        image_path=image_path,
        raw_mask=raw_mask,
        texture_region_map=tex_map,
        boundary_pixels=boundary_pixels,
        mask_out=mask_out,
        overlay_out=overlay_out,
    )

    result = {
        "image_id": image_id,
        "geom_status": "ok",
        "geom_texture_boundary_score": float(texture_boundary_score),
        "geom_object_fraction": float(obj_frac),
        "geom_texture_fraction": float(tex_frac),
        "geom_ambiguous_fraction": float(amb_frac),
        "geom_num_large_texture_regions": int(len(large_ids)),
        "geom_num_strong_boundaries": int(strong_count),
        "geom_boundary_norm": float(boundary_norm),
        "geom_mask_rel": str(mask_out.resolve()),
        "geom_overlay_rel": str(overlay_out.resolve()),
    }

    ensure_dir(cache_meta_path.parent)
    cache_meta_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def enrich_with_sa1b_geometry_and_assets(df: pd.DataFrame, batch_dir: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    masks_dir = ensure_dir(batch_dir / "review" / "masks")
    overlays_dir = ensure_dir(batch_dir / "review" / "overlays")
    meta_dir = ensure_dir(batch_dir / "review" / "meta")

    rows: list[dict[str, Any]] = []
    for _, r in tqdm(out.iterrows(), total=len(out), desc="stage_geom", unit="img"):
        image_id = str(r.get("image_id", ""))
        image_path = Path(str(r.get("image_path", "")))
        ann_ref = str(r.get("annotation_ref", "") or "")
        if not image_id or not image_path.exists() or not ann_ref:
            rows.append(
                {
                    "image_id": image_id,
                    "geom_status": "missing_input",
                    "geom_texture_boundary_score": None,
                    "geom_object_fraction": None,
                    "geom_texture_fraction": None,
                    "geom_ambiguous_fraction": None,
                    "geom_num_large_texture_regions": None,
                    "geom_num_strong_boundaries": None,
                    "geom_boundary_norm": None,
                    "geom_mask_rel": None,
                    "geom_overlay_rel": None,
                }
            )
            continue

        row = _analyze_one(
            image_id=image_id,
            image_path=image_path,
            annotation_ref=ann_ref,
            cache_meta_path=meta_dir / f"{image_id}.json",
            mask_out=masks_dir / f"{image_id}.jpg",
            overlay_out=overlays_dir / f"{image_id}.jpg",
            cfg=cfg,
        )
        rows.append(row)

    geom_df = pd.DataFrame(rows)
    out = out.merge(geom_df, on="image_id", how="left")
    return out
