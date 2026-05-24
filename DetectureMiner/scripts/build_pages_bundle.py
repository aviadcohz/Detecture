#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, ImageFilter


def _balanced_pick(rows: list[dict[str, Any]], max_samples: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= max_samples:
        return rows

    random.seed(seed)
    groups: dict[str, list[dict[str, Any]]] = {"selected": [], "borderline": [], "rejected": []}
    for r in rows:
        st = str(r.get("status") or "rejected")
        if st not in groups:
            st = "rejected"
        groups[st].append(r)

    target = {
        "selected": max(1, int(round(max_samples * 0.40))),
        "borderline": max(1, int(round(max_samples * 0.20))),
        "rejected": max(1, int(round(max_samples * 0.40))),
    }

    picked: list[dict[str, Any]] = []
    def _rank_key(x: dict[str, Any]) -> tuple[int, int, int, float, float]:
        # Prefer rows with explicit overlay-VLM validation signals first.
        has_overlay_vlm = int(x.get("stageD_overlay_score_0_100") is not None)
        has_decision = int(x.get("stageD_decision") is not None)
        has_vlm = int(has_overlay_vlm or has_decision or (x.get("stageD_score_0_100") is not None))
        overlay_score = float(x.get("stageD_overlay_score_0_100") or -1.0)
        return (has_overlay_vlm, has_decision, has_vlm, overlay_score, float(x.get("final_score") or 0.0))

    for k in ("selected", "borderline", "rejected"):
        g = groups[k]
        if not g:
            continue
        g_sorted = sorted(g, key=_rank_key, reverse=True)
        n = min(target[k], len(g_sorted))
        picked.extend(g_sorted[:n])

    if len(picked) < max_samples:
        picked_ids = {str(x.get("image_id")) for x in picked}
        pool = [r for r in rows if str(r.get("image_id")) not in picked_ids]
        pool = sorted(pool, key=_rank_key, reverse=True)
        picked.extend(pool[: max_samples - len(picked)])

    if len(picked) > max_samples:
        picked = sorted(picked, key=_rank_key, reverse=True)[:max_samples]

    return picked


def _safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _make_proxy_mask(src: Path, dst: Path) -> bool:
    try:
        with Image.open(src) as im:
            gray = im.convert("L")
            edges = ImageOps.autocontrast(gray.filter(ImageFilter.FIND_EDGES))
            mask = edges.point(lambda p: 255 if p >= 36 else 0, mode="L")
            out = ImageOps.colorize(mask, black=(8, 8, 8), white=(245, 245, 245)).convert("RGB")
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst, quality=90)
        return True
    except Exception:
        return False


def _make_proxy_overlay(src: Path, dst: Path) -> bool:
    try:
        with Image.open(src) as im:
            base = im.convert("RGB")
            edges = ImageOps.autocontrast(base.convert("L").filter(ImageFilter.FIND_EDGES))
            edge_mask = edges.point(lambda p: min(255, int(p * 1.9)), mode="L")
            tint = Image.new("RGB", base.size, (255, 90, 20))
            comp = Image.composite(tint, base, edge_mask)
            out = Image.blend(base, comp, 0.42)
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst, quality=90)
        return True
    except Exception:
        return False


def _safe_slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text).strip())
    s = s.strip("._")
    return s or "item"


def _row_key(row: dict[str, Any]) -> str:
    dataset = str(row.get("dataset") or "unknown").strip()
    image_id = str(row.get("image_id") or "unknown").strip()
    return f"{dataset}::{image_id}"


def _load_existing_rows(data_json_path: Path) -> list[dict[str, Any]]:
    if not data_json_path.exists():
        return []
    try:
        rows = json.loads(data_json_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for r in rows:
        if isinstance(r, dict):
            out.append(dict(r))
    return out


def build_bundle(
    source_review_dir: Path,
    out_review_dir: Path,
    max_samples: int,
    seed: int,
    dataset_id: str = "",
    merge_existing: bool = False,
    replace_dataset: bool = False,
) -> dict[str, Any]:
    data_path = source_review_dir / "data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data.json: {data_path}")

    rows = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError("data.json is not a list")

    dataset_id = str(dataset_id or "").strip()
    if dataset_id:
        for r in rows:
            if isinstance(r, dict):
                r["dataset"] = dataset_id

    if int(max_samples) <= 0:
        selected = list(rows)
    else:
        selected = _balanced_pick(rows, max_samples=max_samples, seed=seed)

    existing_rows = _load_existing_rows(out_review_dir / "data.json") if merge_existing else []
    replaced_existing = 0
    if merge_existing and replace_dataset and dataset_id:
        kept_rows = [r for r in existing_rows if str(r.get("dataset") or "").strip() != dataset_id]
        replaced_existing = len(existing_rows) - len(kept_rows)
        existing_rows = kept_rows
    merged: dict[str, dict[str, Any]] = {}
    for r in existing_rows:
        merged[_row_key(r)] = dict(r)

    out_review_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("thumbnails", "masks", "overlays", "originals"):
        (out_review_dir / sub).mkdir(parents=True, exist_ok=True)

    # Copy viewer template and cache-bust data.js to avoid stale browser payloads.
    src_index = source_review_dir / "index.html"
    if not src_index.exists():
        raise FileNotFoundError(f"Missing index.html: {src_index}")
    index_text = src_index.read_text(encoding="utf-8")
    cache_bust = str(int(time.time()))
    index_text = re.sub(
        r'''src=(["'])data\.js\1''',
        f'src="data.js?v={cache_bust}"',
        index_text,
    )
    (out_review_dir / "index.html").write_text(index_text, encoding="utf-8")

    out_rows: list[dict[str, Any]] = []
    copied = {"thumb": 0, "mask": 0, "overlay": 0, "original": 0}

    for r in selected:
        row = dict(r)
        image_id = str(row.get("image_id") or "unknown")
        dataset = str(row.get("dataset") or dataset_id or "unknown")
        asset_stem = _safe_slug(f"{dataset}__{image_id}")
        thumb_out_rel = ""
        orig_out_rel = ""

        image_src_raw = str(row.get("image_path") or "").strip()
        image_src_path: Path | None = None
        if image_src_raw:
            cand = Path(image_src_raw).expanduser()
            if not cand.is_absolute():
                cand2 = source_review_dir / image_src_raw
                if cand2.exists():
                    cand = cand2
            if cand.exists() and cand.is_file():
                image_src_path = cand

        # Original image path can be absolute or relative; copy into bundle for GitHub Pages.
        if image_src_path is not None:
            ext = image_src_path.suffix.lower() or ".jpg"
            dst_orig = out_review_dir / "originals" / f"{asset_stem}{ext}"
            if _safe_copy(image_src_path, dst_orig):
                orig_out_rel = f"originals/{asset_stem}{ext}"
                row["image_path"] = orig_out_rel
                copied["original"] += 1

        # Thumbnail (relative path from source review dir).
        thumb_rel = str(row.get("thumb") or "").strip()
        if thumb_rel:
            src_thumb = source_review_dir / thumb_rel
            thumb_name = f"{asset_stem}.jpg"
            dst_thumb = out_review_dir / "thumbnails" / thumb_name
            if _safe_copy(src_thumb, dst_thumb):
                thumb_out_rel = f"thumbnails/{thumb_name}"
                row["thumb"] = thumb_out_rel
                copied["thumb"] += 1

        # Mask and overlay are already relative in current review site.
        mask_rel = str(row.get("mask_rel") or "").strip()
        mask_ok = False
        if mask_rel:
            src_mask = source_review_dir / mask_rel
            mask_name = f"{asset_stem}.jpg"
            dst_mask = out_review_dir / "masks" / mask_name
            if _safe_copy(src_mask, dst_mask):
                row["mask_rel"] = f"masks/{mask_name}"
                copied["mask"] += 1
                mask_ok = True
        if not mask_ok:
            dst_mask = out_review_dir / "masks" / f"{asset_stem}.jpg"
            if image_src_path is not None and _make_proxy_mask(image_src_path, dst_mask):
                row["mask_rel"] = f"masks/{asset_stem}.jpg"
                copied["mask"] += 1
                mask_ok = True
        if not mask_ok:
            row["mask_rel"] = orig_out_rel or thumb_out_rel or str(row.get("image_path") or "")

        overlay_rel = str(row.get("overlay_rel") or "").strip()
        overlay_ok = False
        if overlay_rel:
            src_overlay = source_review_dir / overlay_rel
            overlay_name = f"{asset_stem}.jpg"
            dst_overlay = out_review_dir / "overlays" / overlay_name
            if _safe_copy(src_overlay, dst_overlay):
                row["overlay_rel"] = f"overlays/{overlay_name}"
                copied["overlay"] += 1
                overlay_ok = True
        if not overlay_ok:
            dst_overlay = out_review_dir / "overlays" / f"{asset_stem}.jpg"
            if image_src_path is not None and _make_proxy_overlay(image_src_path, dst_overlay):
                row["overlay_rel"] = f"overlays/{asset_stem}.jpg"
                copied["overlay"] += 1
                overlay_ok = True
        if not overlay_ok:
            row["overlay_rel"] = orig_out_rel or thumb_out_rel or str(row.get("image_path") or "")

        if not str(row.get("image_path") or "").strip():
            row["image_path"] = thumb_out_rel

        merged[_row_key(row)] = row

    out_rows.extend(merged.values())
    out_rows.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)

    # Write JSON + JS payload for viewer.
    data_json_path = out_review_dir / "data.json"
    data_js_path = out_review_dir / "data.js"
    data_json_path.write_text(json.dumps(out_rows, ensure_ascii=False), encoding="utf-8")
    data_js_path.write_text("window.REVIEW_DATA = " + json.dumps(out_rows, ensure_ascii=False) + ";\n", encoding="utf-8")

    by_status: dict[str, int] = {}
    by_dataset: dict[str, int] = {}
    for r in out_rows:
        st = str(r.get("status") or "")
        ds = str(r.get("dataset") or "")
        by_status[st] = by_status.get(st, 0) + 1
        by_dataset[ds] = by_dataset.get(ds, 0) + 1

    summary = {
        "source": str(source_review_dir),
        "out": str(out_review_dir),
        "n_input": len(rows),
        "n_existing": len(existing_rows),
        "n_existing_replaced_for_dataset": int(replaced_existing),
        "n_new_selected": len(selected),
        "n_output": len(out_rows),
        "copied": copied,
        "by_status": by_status,
        "by_dataset": by_dataset,
    }
    (out_review_dir / "bundle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Build compact GitHub Pages bundle from local review site")
    ap.add_argument("--source_review", required=True)
    ap.add_argument("--out_review", required=True)
    ap.add_argument("--max_samples", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset_id", default="")
    ap.add_argument("--merge_existing", action="store_true")
    ap.add_argument("--replace_dataset", action="store_true")
    args = ap.parse_args()

    summary = build_bundle(
        source_review_dir=Path(args.source_review),
        out_review_dir=Path(args.out_review),
        max_samples=int(args.max_samples),
        seed=int(args.seed),
        dataset_id=str(args.dataset_id),
        merge_existing=bool(args.merge_existing),
        replace_dataset=bool(args.replace_dataset),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
