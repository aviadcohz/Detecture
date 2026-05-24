from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

PVA_DEFAULT_NAMES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}

BUILT_STRUCTURE_DEFAULT_NAMES = {
    "building-other-merged",
    "roof",
    "wall-other-merged",
    "wall-brick",
    "wall-concrete",
    "bridge",
    "tent",
}


def _as_image_id_keys(value: Any) -> list[str]:
    s = str(value).strip()
    if not s:
        return []
    out = [s]
    if s.isdigit():
        out.append(f"{int(s):012d}")
    return out


def _extract_annotation_path(annotation_ref: Any) -> Path | None:
    if annotation_ref is None:
        return None
    s = str(annotation_ref).strip()
    if not s:
        return None
    if "::" in s:
        s = s.split("::", 1)[0]
    p = Path(s)
    return p if p.exists() else None


def _infer_panoptic_json_path(df: pd.DataFrame, cfg: dict[str, Any]) -> Path | None:
    explicit = str(cfg.get("panoptic_json") or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p

    refs = df.get("annotation_ref")
    if refs is None:
        return None

    seen: set[Path] = set()
    for ref in refs.tolist():
        ann_path = _extract_annotation_path(ref)
        if ann_path is None:
            continue
        if ann_path in seen:
            continue
        seen.add(ann_path)

        name = ann_path.name.lower()
        cands: list[Path] = []
        if "val2017" in name:
            cands.append(ann_path.with_name("panoptic_val2017.json"))
        if "train2017" in name:
            cands.append(ann_path.with_name("panoptic_train2017.json"))
        if name.startswith("stuff_"):
            cands.append(ann_path.with_name(name.replace("stuff_", "panoptic_", 1)))
        cands.append(ann_path.with_name("panoptic_val2017.json"))
        cands.append(ann_path.with_name("panoptic_train2017.json"))
        for c in cands:
            if c.exists():
                return c
    return None


def _build_panoptic_index(panoptic_json: Path) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    payload = json.loads(panoptic_json.read_text(encoding="utf-8"))
    anns = payload.get("annotations", [])
    cats = payload.get("categories", [])
    cat_by_id: dict[int, dict[str, Any]] = {}
    for c in cats:
        if isinstance(c, dict) and c.get("id") is not None:
            cat_by_id[int(c["id"])] = c

    ann_by_key: dict[str, dict[str, Any]] = {}
    for ann in anns:
        if not isinstance(ann, dict):
            continue
        keys: list[str] = []
        file_name = ann.get("file_name")
        if file_name:
            keys.append(Path(str(file_name)).stem)
        image_id = ann.get("image_id")
        if image_id is not None:
            sid = str(image_id)
            keys.append(sid)
            if sid.isdigit():
                keys.append(f"{int(sid):012d}")
        for k in keys:
            ann_by_key.setdefault(k, ann)
    return ann_by_key, cat_by_id


def enrich_with_coco_panoptic_metrics(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["thing_fraction"] = pd.NA
    out["largest_thing_fraction"] = pd.NA
    out["num_large_thing_instances"] = pd.NA
    out["person_vehicle_animal_fraction"] = pd.NA
    out["sky_fraction"] = pd.NA
    out["built_structure_fraction"] = pd.NA
    out["semantic_metrics_status"] = "disabled"

    if not bool(cfg.get("enabled", False)):
        return out

    panoptic_json = _infer_panoptic_json_path(out, cfg)
    if panoptic_json is None:
        out["semantic_metrics_status"] = "missing_panoptic_json"
        return out

    try:
        ann_by_key, cat_by_id = _build_panoptic_index(panoptic_json)
    except Exception:
        out["semantic_metrics_status"] = "panoptic_load_error"
        return out

    large_inst_min = float(cfg.get("large_thing_instance_min_fraction", 0.005))
    pva_names = {str(x).strip().lower() for x in cfg.get("person_vehicle_animal_names", sorted(PVA_DEFAULT_NAMES))}
    built_names = {str(x).strip().lower() for x in cfg.get("built_structure_names", sorted(BUILT_STRUCTURE_DEFAULT_NAMES))}

    statuses: list[str] = []
    tf_col: list[float | None] = []
    ltf_col: list[float | None] = []
    nlarge_col: list[int | None] = []
    pva_col: list[float | None] = []
    sky_col: list[float | None] = []
    built_col: list[float | None] = []

    for _, row in out.iterrows():
        keys = _as_image_id_keys(row.get("image_id"))
        image_path = Path(str(row.get("image_path", "")))
        stem = image_path.stem
        if stem:
            keys.append(stem)

        ann = None
        for k in keys:
            ann = ann_by_key.get(k)
            if ann is not None:
                break

        if ann is None:
            statuses.append("missing_image_in_panoptic")
            tf_col.append(None)
            ltf_col.append(None)
            nlarge_col.append(None)
            pva_col.append(None)
            sky_col.append(None)
            built_col.append(None)
            continue

        segments = ann.get("segments_info", [])
        if not isinstance(segments, list) or not segments:
            statuses.append("missing_segments")
            tf_col.append(0.0)
            ltf_col.append(0.0)
            nlarge_col.append(0)
            pva_col.append(0.0)
            sky_col.append(0.0)
            built_col.append(0.0)
            continue

        total_area = float(sum(float(s.get("area", 0.0)) for s in segments if isinstance(s, dict)))
        if total_area <= 0.0:
            statuses.append("zero_total_area")
            tf_col.append(0.0)
            ltf_col.append(0.0)
            nlarge_col.append(0)
            pva_col.append(0.0)
            sky_col.append(0.0)
            built_col.append(0.0)
            continue

        thing_areas: list[float] = []
        pva_area = 0.0
        sky_area = 0.0
        built_area = 0.0
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            area = float(seg.get("area", 0.0))
            if area <= 0.0:
                continue
            cat_id = seg.get("category_id")
            if cat_id is None:
                continue
            cat = cat_by_id.get(int(cat_id), {})
            cat_name = str(cat.get("name", "")).strip().lower()
            if "sky" in cat_name:
                sky_area += area
            if cat_name in built_names or "building" in cat_name:
                built_area += area
            isthing = int(cat.get("isthing", 0)) == 1
            if not isthing:
                continue
            thing_areas.append(area)
            if cat_name in pva_names:
                pva_area += area

        if not thing_areas:
            statuses.append("ok")
            tf_col.append(0.0)
            ltf_col.append(0.0)
            nlarge_col.append(0)
            pva_col.append(0.0)
            sky_col.append(float(sky_area / total_area))
            built_col.append(float(built_area / total_area))
            continue

        thing_fraction = float(sum(thing_areas) / total_area)
        largest_thing_fraction = float(max(thing_areas) / total_area)
        num_large = int(sum(1 for a in thing_areas if (a / total_area) > large_inst_min))
        pva_fraction = float(pva_area / total_area)
        sky_fraction = float(sky_area / total_area)
        built_fraction = float(built_area / total_area)

        statuses.append("ok")
        tf_col.append(thing_fraction)
        ltf_col.append(largest_thing_fraction)
        nlarge_col.append(num_large)
        pva_col.append(pva_fraction)
        sky_col.append(sky_fraction)
        built_col.append(built_fraction)

    out["thing_fraction"] = tf_col
    out["largest_thing_fraction"] = ltf_col
    out["num_large_thing_instances"] = nlarge_col
    out["person_vehicle_animal_fraction"] = pva_col
    out["sky_fraction"] = sky_col
    out["built_structure_fraction"] = built_col
    out["semantic_metrics_status"] = statuses
    return out
