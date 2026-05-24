from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps

from rwtd_miner.utils.io import ensure_dir


def _make_thumb(src: Path, dst: Path, size: tuple[int, int] = (300, 220)) -> None:
    try:
        with Image.open(src) as im:
            tile = ImageOps.fit(im.convert("RGB"), size, method=Image.Resampling.BICUBIC)
        ensure_dir(dst.parent)
        tile.save(dst, quality=88)
    except Exception:
        pass


def _localize_asset(src_value: Any, fallback_name: str, dst_dir: Path, rel_prefix: str) -> str | None:
    if src_value is None or (isinstance(src_value, float) and pd.isna(src_value)):
        return None
    src_raw = str(src_value).strip()
    if not src_raw:
        return None

    src_path = Path(src_raw)
    if not src_path.is_absolute():
        # Keep already-relative paths as-is.
        return src_raw

    if not src_path.exists():
        return None

    suffix = src_path.suffix.lower() or ".jpg"
    dst_path = dst_dir / f"{fallback_name}{suffix}"
    if src_path.resolve() == dst_path.resolve():
        return f"{rel_prefix}/{dst_path.name}"

    try:
        ensure_dir(dst_path.parent)
        if dst_path.exists():
            dst_path.unlink()
        shutil.copy2(src_path, dst_path)
        return f"{rel_prefix}/{dst_path.name}"
    except Exception:
        return None


def _final_score(row: pd.Series) -> float:
    stage_d = row.get("stageD_score_0_100")
    stage_b = row.get("stageB_clip_score")
    stage_a = row.get("stageA_rwtd_score")
    review_score = row.get("review_score")

    if pd.notna(review_score):
        return float(review_score)
    if pd.notna(stage_d):
        return float(stage_d)
    if pd.notna(stage_b):
        return float(stage_b) * 100.0
    if pd.notna(stage_a):
        return float(stage_a)
    return 0.0


def _load_review_template() -> str:
    template_path = Path(__file__).with_name("review_template.html")
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"/><title>RWTD Review</title></head>
<body>
  <h2>RWTD Review</h2>
  <p>Missing review template file: <code>rwtd_miner/utils/review_template.html</code></p>
  <script src="data.js"></script>
  <pre id="out"></pre>
  <script>
    document.getElementById("out").textContent = JSON.stringify(window.REVIEW_DATA || [], null, 2);
  </script>
</body>
</html>
"""


def build_review_site(df: pd.DataFrame, batch_dir: Path) -> Path:
    review_dir = ensure_dir(batch_dir / "review")
    thumbs_dir = ensure_dir(review_dir / "thumbnails")
    masks_dir = ensure_dir(review_dir / "masks")
    overlays_dir = ensure_dir(review_dir / "overlays")

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        image_path = Path(str(r.get("image_path", "")))
        if not image_path.exists():
            continue

        image_id = str(r.get("image_id", ""))

        thumb_rel = f"thumbnails/{image_id}.jpg"
        thumb_abs = thumbs_dir / f"{image_id}.jpg"
        _make_thumb(image_path, thumb_abs)

        mask_rel = _localize_asset(r.get("geom_mask_rel"), image_id, masks_dir, "masks")
        overlay_rel = _localize_asset(r.get("geom_overlay_rel"), image_id, overlays_dir, "overlays")

        status = "rejected"
        if bool(r.get("final_selected", False)):
            status = "selected"
        elif bool(r.get("final_borderline", False)):
            status = "borderline"

        rows.append(
            {
                "image_id": image_id,
                "dataset": "" if pd.isna(r.get("dataset")) else str(r.get("dataset")),
                "image_path": str(image_path),
                "thumb": thumb_rel,
                "status": status,
                "final_score": round(_final_score(r), 3),
                "selection_reason": "" if pd.isna(r.get("selection_reason")) else str(r.get("selection_reason")),
                "score_breakdown_json": "" if pd.isna(r.get("score_breakdown_json")) else str(r.get("score_breakdown_json")),
                "stageA_rwtd_score": None if pd.isna(r.get("stageA_rwtd_score")) else round(float(r.get("stageA_rwtd_score")), 3),
                "stageB_clip_score": None if pd.isna(r.get("stageB_clip_score")) else round(float(r.get("stageB_clip_score")), 6),
                "stageD_score_0_100": None
                if pd.isna(r.get("stageD_score_0_100"))
                else round(float(r.get("stageD_score_0_100")), 3),
                "stageA_n_masks": None if pd.isna(r.get("stageA_n_masks")) else int(r.get("stageA_n_masks")),
                "stageA_largest_ratio": None
                if pd.isna(r.get("stageA_largest_ratio"))
                else round(float(r.get("stageA_largest_ratio")), 6),
                "stageA_small_frac": None
                if pd.isna(r.get("stageA_small_frac"))
                else round(float(r.get("stageA_small_frac")), 6),
                "stageA_pass": bool(r.get("stageA_pass", False)),
                "stageB_pass": bool(r.get("stageB_pass", False)),
                "stageC_pass": None if pd.isna(r.get("stageC_pass")) else bool(r.get("stageC_pass")),
                "stageD_decision": None if pd.isna(r.get("stageD_decision")) else str(r.get("stageD_decision")),
                "stageD_overlay_score_0_100": None
                if pd.isna(r.get("stageD_overlay_score_0_100"))
                else round(float(r.get("stageD_overlay_score_0_100")), 3),
                "stageD_overlay_pass": None if pd.isna(r.get("stageD_overlay_pass")) else bool(r.get("stageD_overlay_pass")),
                "stageD_overlay_reason": None if pd.isna(r.get("stageD_overlay_reason")) else str(r.get("stageD_overlay_reason")),
                "geom_texture_boundary_score": None
                if pd.isna(r.get("geom_texture_boundary_score"))
                else round(float(r.get("geom_texture_boundary_score")), 3),
                "geom_object_fraction": None
                if pd.isna(r.get("geom_object_fraction"))
                else round(float(r.get("geom_object_fraction")), 6),
                "geom_texture_fraction": None
                if pd.isna(r.get("geom_texture_fraction"))
                else round(float(r.get("geom_texture_fraction")), 6),
                "geom_num_large_texture_regions": None
                if pd.isna(r.get("geom_num_large_texture_regions"))
                else int(r.get("geom_num_large_texture_regions")),
                "geom_num_strong_boundaries": None
                if pd.isna(r.get("geom_num_strong_boundaries"))
                else int(r.get("geom_num_strong_boundaries")),
                "geom_boundary_norm": None if pd.isna(r.get("geom_boundary_norm")) else round(float(r.get("geom_boundary_norm")), 6),
                "mask_rel": mask_rel,
                "overlay_rel": overlay_rel,
            }
        )

    rows.sort(key=lambda x: x["final_score"], reverse=True)

    (review_dir / "data.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    (review_dir / "data.js").write_text(
        f"window.REVIEW_DATA = {json.dumps(rows, ensure_ascii=False)};\n",
        encoding="utf-8",
    )
    (review_dir / "index.html").write_text(_load_review_template(), encoding="utf-8")
    return review_dir / "index.html"
