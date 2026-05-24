from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.io import loadmat
except Exception:  # pragma: no cover
    loadmat = None


def _segments_from_label_map(
    label_map: np.ndarray,
    min_area_px: int = 64,
    min_area_frac: float = 0.0006,
) -> dict[str, Any]:
    if label_map.ndim != 2:
        raise ValueError("Label map must be 2D")
    h, w = label_map.shape
    lm = np.asarray(label_map)
    min_area = max(int(min_area_px), int(round(float(min_area_frac) * float(h * w))))
    anns: list[dict[str, Any]] = []
    labels = np.unique(lm)
    for lab in labels.tolist():
        lab_i = int(lab)
        if lab_i <= 0:
            continue
        mask = lm == lab
        area = int(mask.sum())
        if area < min_area:
            continue
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        x0 = int(xs.min())
        x1 = int(xs.max())
        y0 = int(ys.min())
        y1 = int(ys.max())
        bw = int(x1 - x0 + 1)
        bh = int(y1 - y0 + 1)
        anns.append(
            {
                "area": float(area),
                "bbox": [x0, y0, bw, bh],
                "category_id": lab_i,
                # Keep raw binary segmentation for downstream geometry rendering.
                "segmentation": mask.astype(np.uint8),
            }
        )
    return {
        "image": {"width": int(w), "height": int(h)},
        "annotations": anns,
        "__rwtd_source": "dense_label_map",
        "__rwtd_min_area_px": int(min_area),
    }


def load_mat_as_annotation_payload(path: Path, min_area_px: int = 64) -> dict[str, Any] | None:
    if loadmat is None:
        return None
    p = Path(path)
    if p.suffix.lower() != ".mat" or (not p.exists()):
        return None

    mat = loadmat(str(p), squeeze_me=True, struct_as_record=False)
    if "LabelMap" in mat:
        lm = np.asarray(mat["LabelMap"])
        return _segments_from_label_map(lm, min_area_px=min_area_px)

    if "groundTruth" in mat:
        gt = np.atleast_1d(mat["groundTruth"])
        seg_maps: list[np.ndarray] = []
        for entry in gt.tolist():
            seg = getattr(entry, "Segmentation", None)
            if seg is None:
                continue
            seg_arr = np.asarray(seg)
            if seg_arr.ndim == 2 and seg_arr.size > 0:
                seg_maps.append(seg_arr)
        if not seg_maps:
            return None
        # Prefer the lower-quartile granularity so boundaries are coarse/stable.
        seg_maps.sort(key=lambda x: int(np.unique(x).size))
        q1_idx = max(0, min(len(seg_maps) - 1, int(round((len(seg_maps) - 1) * 0.25))))
        chosen = seg_maps[q1_idx]
        return _segments_from_label_map(chosen, min_area_px=min_area_px)

    return None
