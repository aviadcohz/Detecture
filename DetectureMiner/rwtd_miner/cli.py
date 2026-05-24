from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from rwtd_miner.dataset_adapters.local_sa1b import LocalSA1BAdapter
from rwtd_miner.utils.ade20k_reference import (
    build_ade20k_class_map,
    ensure_ade20k_dataset,
    render_ade20k_review_assets,
    run_ade20k_full_eval,
)
from rwtd_miner.utils.checkpoint import Checkpoint
from rwtd_miner.utils.image_utils import make_contact_sheet, sample_paths, score_histogram, symlink_or_copy
from rwtd_miner.utils.io import ensure_dir, read_table, write_json, write_table
from rwtd_miner.utils.logging import configure_logging, get_logger
from rwtd_miner.utils.review_site import build_review_site
from rwtd_miner.utils.rwtd_reference import (
    apply_texture_priority_scoring,
    fetch_rwtd_from_texturesam,
    score_rwtd_reference_dataset,
)
from rwtd_miner.utils.sa1b_geometry import enrich_with_sa1b_geometry_and_assets
from rwtd_miner.utils.coco_semantic import enrich_with_coco_panoptic_metrics
from rwtd_miner.utils.sa1b_sample_fetch import fetch_sa1b_pairs_from_tar


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _budget_to_bytes(gb: float) -> int:
    return int(float(gb) * (1024**3))


def _assign_batches(df: pd.DataFrame, budget_bytes: int) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["batch_id"] = []
        return out

    df = df.sort_values("image_path").reset_index(drop=True)
    batch_ids: list[int] = []
    cur_batch = 0
    cur_bytes = 0

    for _, row in df.iterrows():
        sz = int(row["file_size_bytes"])
        if cur_bytes > 0 and cur_bytes + sz > budget_bytes:
            cur_batch += 1
            cur_bytes = 0
        batch_ids.append(cur_batch)
        cur_bytes += sz

    out = df.copy()
    out["batch_id"] = batch_ids
    return out


def _build_image_index(
    input_root: Path,
    out_root: Path,
    cfg: dict[str, Any],
    adapter_name: str = "local",
    tfds_name: str = "segment_anything",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = get_logger("index")
    if adapter_name == "local":
        adapter = LocalSA1BAdapter(input_root=input_root, config=cfg)
    elif adapter_name == "tfds":
        from rwtd_miner.dataset_adapters.tfds_sa1b import TFDSSA1BAdapter

        adapter = TFDSSA1BAdapter(input_root=input_root, config=cfg, dataset_name=tfds_name)
    else:
        raise ValueError(f"Unsupported adapter: {adapter_name}")

    records = []
    for rec in adapter.discover_images():
        records.append(
            {
                "image_id": rec.image_id,
                "image_path": str(rec.image_path),
                "annotation_ref": rec.annotation_ref,
                "file_size_bytes": int(rec.file_size_bytes),
                "width": rec.width,
                "height": rec.height,
                "dataset": adapter.dataset_name(),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No images discovered")

    idx_cfg = cfg.get("index", {})
    min_short_side = int(idx_cfg.get("min_short_side", 256))
    short_side = np.minimum(df["width"].fillna(0).astype(int), df["height"].fillna(0).astype(int))
    df["resolution_ok"] = short_side >= min_short_side

    budget_gb = float(cfg.get("batching", {}).get("batch_budget_gb", 5.0))
    assigned = _assign_batches(df, _budget_to_bytes(budget_gb))

    index_dir = ensure_dir(out_root / "index")
    write_table(df, index_dir / "image_index", write_csv=True)
    write_table(assigned[["image_id", "batch_id", "file_size_bytes", "image_path"]], index_dir / "batch_assignments", write_csv=True)

    log.info("Indexed %s images into %s batches", len(df), int(assigned["batch_id"].max()) + 1)
    return df, assigned


def _load_index_or_build(
    input_root: Path,
    out_root: Path,
    cfg: dict[str, Any],
    adapter_name: str = "local",
    tfds_name: str = "segment_anything",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_df = read_table(out_root / "index" / "image_index")
    batch_df = read_table(out_root / "index" / "batch_assignments")
    if index_df.empty or batch_df.empty:
        return _build_image_index(
            input_root=input_root,
            out_root=out_root,
            cfg=cfg,
            adapter_name=adapter_name,
            tfds_name=tfds_name,
        )
    merged = index_df.merge(batch_df[["image_id", "batch_id"]], on="image_id", how="left")
    return index_df, merged


def _save_batch_manifest(df: pd.DataFrame, batch_dir: Path, write_csv: bool) -> None:
    write_table(df, batch_dir / "batch_manifest", write_csv=write_csv)


def _apply_semantic_object_gates(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    gate_cfg = cfg.get("semantic_gates", {})
    out["semantic_gate_failed"] = False
    out["semantic_gate_reasons"] = ""

    if not bool(gate_cfg.get("enabled", False)):
        return out

    dataset_mask = pd.Series(True, index=out.index)
    target_datasets = {str(x).strip() for x in gate_cfg.get("target_datasets", []) if str(x).strip()}
    if target_datasets and "dataset" in out.columns:
        dataset_vals = out["dataset"].fillna("").astype(str)
        dataset_mask = dataset_vals.isin(target_datasets)

    tf = pd.to_numeric(out.get("thing_fraction", pd.Series([np.nan] * len(out))), errors="coerce")
    ltf = pd.to_numeric(out.get("largest_thing_fraction", pd.Series([np.nan] * len(out))), errors="coerce")
    nlarge = pd.to_numeric(out.get("num_large_thing_instances", pd.Series([np.nan] * len(out))), errors="coerce")
    pva = pd.to_numeric(out.get("person_vehicle_animal_fraction", pd.Series([np.nan] * len(out))), errors="coerce")

    fail = pd.Series(False, index=out.index)
    reasons: list[list[str]] = [[] for _ in range(len(out))]

    def mark(mask: pd.Series, reason: str) -> None:
        nonlocal fail
        m = mask.fillna(False)
        if not bool(m.any()):
            return
        fail = fail | m
        idxs = np.where(m.to_numpy())[0]
        for i in idxs:
            if reason not in reasons[i]:
                reasons[i].append(reason)

    if bool(gate_cfg.get("missing_metrics_as_fail", False)):
        missing = dataset_mask & (tf.isna() | ltf.isna() | nlarge.isna() | pva.isna())
        mark(missing, "missing_semantic_metrics")

    if gate_cfg.get("thing_fraction_max", None) not in (None, ""):
        thr = float(gate_cfg["thing_fraction_max"])
        mark(dataset_mask & (tf > thr), "thing_fraction")

    if gate_cfg.get("largest_thing_fraction_max", None) not in (None, ""):
        thr = float(gate_cfg["largest_thing_fraction_max"])
        mark(dataset_mask & (ltf > thr), "largest_thing_fraction")

    if gate_cfg.get("num_large_thing_instances_max", None) not in (None, ""):
        thr = float(gate_cfg["num_large_thing_instances_max"])
        mark(dataset_mask & (nlarge > thr), "num_large_thing_instances")

    if gate_cfg.get("person_vehicle_animal_fraction_max", None) not in (None, ""):
        thr = float(gate_cfg["person_vehicle_animal_fraction_max"])
        mark(dataset_mask & (pva > thr), "person_vehicle_animal_fraction")

    out["semantic_gate_failed"] = fail.astype(bool)
    out["semantic_gate_reasons"] = ["|".join(xs) if xs else "" for xs in reasons]
    if bool(gate_cfg.get("apply_as_hard_reject", False)):
        out.loc[out["semantic_gate_failed"], "final_selected"] = False
        out.loc[out["semantic_gate_failed"], "final_borderline"] = False
    if "selection_reason" in out.columns:
        gate_rows = out["semantic_gate_failed"].fillna(False)
        out.loc[gate_rows, "selection_reason"] = (
            out.loc[gate_rows, "selection_reason"]
            .fillna("")
            .astype(str)
            .map(lambda s: s if "semantic_gate" in s else (f"{s}|semantic_gate" if s else "semantic_gate"))
        )
    return out


def _apply_generalized_selection_model(df: pd.DataFrame, cfg: dict[str, Any], stage_d_enabled: bool) -> pd.DataFrame:
    out = df.copy()
    sel_cfg = cfg.get("selection", {})
    model_cfg = sel_cfg.get("generalized_model", {})
    if not bool(model_cfg.get("enabled", True)):
        return out

    n = len(out)
    idx = out.index

    geom = pd.to_numeric(out.get("geom_texture_boundary_score", pd.Series([np.nan] * n, index=idx)), errors="coerce")
    if geom.notna().sum() == 0:
        geom = pd.to_numeric(out.get("stageA_rwtd_score", pd.Series([np.nan] * n, index=idx)), errors="coerce")
    clip_raw = pd.to_numeric(out.get("stageB_clip_score", pd.Series([np.nan] * n, index=idx)), errors="coerce")
    clip100 = _clip_score_to_100(clip_raw)
    vlm = pd.to_numeric(out.get("stageD_score_0_100", pd.Series([np.nan] * n, index=idx)), errors="coerce")

    w_geom = float(model_cfg.get("weight_geometry", 0.62))
    w_clip = float(model_cfg.get("weight_clip", 0.16))
    w_vlm = float(model_cfg.get("weight_vlm", 0.22))

    fused_num = pd.Series(np.zeros(n, dtype=float), index=idx)
    fused_den = pd.Series(np.zeros(n, dtype=float), index=idx)

    m_geom = geom.notna()
    fused_num = fused_num + np.where(m_geom, w_geom * geom.fillna(0.0), 0.0)
    fused_den = fused_den + np.where(m_geom, w_geom, 0.0)

    m_clip = clip_raw.notna()
    fused_num = fused_num + np.where(m_clip, w_clip * clip100.fillna(0.0), 0.0)
    fused_den = fused_den + np.where(m_clip, w_clip, 0.0)

    m_vlm = vlm.notna()
    fused_num = fused_num + np.where(m_vlm, w_vlm * vlm.fillna(0.0), 0.0)
    fused_den = fused_den + np.where(m_vlm, w_vlm, 0.0)

    prior = pd.to_numeric(out.get("review_score", pd.Series([0.0] * n, index=idx)), errors="coerce").fillna(0.0)
    fused = fused_num / fused_den.replace(0.0, np.nan)
    fused = fused.fillna(prior).astype(float)

    tf = pd.to_numeric(out.get("thing_fraction", pd.Series([np.nan] * n, index=idx)), errors="coerce").fillna(0.0)
    ltf = pd.to_numeric(out.get("largest_thing_fraction", pd.Series([np.nan] * n, index=idx)), errors="coerce").fillna(0.0)
    pva = pd.to_numeric(out.get("person_vehicle_animal_fraction", pd.Series([np.nan] * n, index=idx)), errors="coerce").fillna(0.0)
    nlarge = pd.to_numeric(out.get("num_large_thing_instances", pd.Series([np.nan] * n, index=idx)), errors="coerce").fillna(0.0)
    geom_obj = pd.to_numeric(out.get("geom_object_fraction", pd.Series([np.nan] * n, index=idx)), errors="coerce").fillna(0.0)
    sky_frac = pd.to_numeric(out.get("sky_fraction", pd.Series([np.nan] * n, index=idx)), errors="coerce").fillna(0.0)
    built_frac = pd.to_numeric(out.get("built_structure_fraction", pd.Series([np.nan] * n, index=idx)), errors="coerce").fillna(0.0)

    soft_penalty = (
        float(model_cfg.get("soft_penalty_thing_fraction_weight", 28.0)) * tf
        + float(model_cfg.get("soft_penalty_largest_thing_weight", 36.0)) * ltf
        + float(model_cfg.get("soft_penalty_pva_weight", 42.0)) * pva
        + float(model_cfg.get("soft_penalty_geom_object_weight", 16.0)) * geom_obj
        + float(model_cfg.get("soft_penalty_large_instances_weight", 3.0)) * np.clip(nlarge - 1.0, 0.0, None)
        + float(model_cfg.get("soft_penalty_sky_fraction_weight", 0.0)) * sky_frac
        + float(model_cfg.get("soft_penalty_built_structure_weight", 0.0)) * built_frac
    )

    flags = out.get("stageD_flags", pd.Series([""] * n, index=idx)).fillna("").astype(str)
    decisions = out.get("stageD_decision", pd.Series([""] * n, index=idx)).fillna("").astype(str)

    def _flags_has(flag_name: str) -> pd.Series:
        if not flag_name:
            return pd.Series(False, index=idx)
        pat = rf"(?:^|\|){re.escape(str(flag_name))}(?:$|\|)"
        return flags.str.contains(pat, na=False, regex=True)

    hard = pd.Series(False, index=idx)
    if bool(model_cfg.get("hard_veto_enabled", True)):
        hard = hard | (tf > float(model_cfg.get("hard_veto_thing_fraction_max", 0.35)))
        hard = hard | (ltf > float(model_cfg.get("hard_veto_largest_thing_fraction_max", 0.15)))
        hard = hard | (nlarge > float(model_cfg.get("hard_veto_num_large_instances_max", 3)))
        hard = hard | (pva > float(model_cfg.get("hard_veto_pva_fraction_max", 0.08)))
        if model_cfg.get("hard_veto_sky_fraction_max", None) not in (None, ""):
            hard = hard | (sky_frac > float(model_cfg.get("hard_veto_sky_fraction_max")))
        if model_cfg.get("hard_veto_built_structure_fraction_max", None) not in (None, ""):
            hard = hard | (built_frac > float(model_cfg.get("hard_veto_built_structure_fraction_max")))
        if bool(model_cfg.get("hard_veto_object_centric_flag", True)):
            hard = hard | _flags_has("object_centric")
        hard_flags = model_cfg.get("hard_veto_flags", ["salient_semantic_subject"])
        if isinstance(hard_flags, list):
            for f in hard_flags:
                fs = str(f).strip()
                if fs:
                    hard = hard | _flags_has(fs)

    rare_semantic_override = pd.Series(False, index=idx)
    if bool(model_cfg.get("allow_rare_semantic_override", True)):
        sem_flag = str(model_cfg.get("rare_semantic_override_flag", "salient_semantic_subject")).strip()
        sem_rows = _flags_has(sem_flag) | _flags_has("object_centric")
        rare_semantic_override = (
            sem_rows
            & (vlm >= float(model_cfg.get("rare_semantic_override_min_vlm_score", 96.0)))
            & (geom >= float(model_cfg.get("rare_semantic_override_min_geom_score", 97.0)))
            & (geom_obj <= float(model_cfg.get("rare_semantic_override_max_geom_object_fraction", 0.01)))
            & (~_flags_has("too_many_objects"))
        )
        hard = hard & (~rare_semantic_override)

    override = pd.Series(False, index=idx)
    if bool(model_cfg.get("texture_override_enabled", True)):
        override = (
            (geom >= float(model_cfg.get("texture_override_min_geom_score", 88.0)))
            & (tf <= float(model_cfg.get("texture_override_max_thing_fraction", 0.03)))
            & (ltf <= float(model_cfg.get("texture_override_max_largest_thing_fraction", 0.015)))
            & (pva <= float(model_cfg.get("texture_override_max_pva_fraction", 0.01)))
            & (geom_obj <= float(model_cfg.get("texture_override_max_geom_object_fraction", 0.08)))
            & (~flags.str.contains("object_centric", na=False))
        )

    score = fused - soft_penalty + np.where(override, float(model_cfg.get("texture_override_bonus", 18.0)), 0.0)
    score = score.clip(lower=0.0, upper=100.0).astype(float)
    out["review_score"] = score
    out["objectness_soft_penalty"] = soft_penalty.astype(float)
    out["objectness_hard_veto"] = hard.astype(bool)
    out["texture_override_applied"] = override.astype(bool)
    out["rare_semantic_override_applied"] = rare_semantic_override.astype(bool)

    selected_min = float(model_cfg.get("selected_score_min", sel_cfg.get("match_score_min", 80)))
    borderline_min = float(model_cfg.get("borderline_score_min", sel_cfg.get("borderline_score_min", 70)))

    out["final_selected"] = (score >= selected_min) & (~hard)
    out["final_borderline"] = (~out["final_selected"]) & (score >= borderline_min) & (~hard)

    if bool(model_cfg.get("require_match_decision_for_selected", False)):
        # Apply this only where VLM is actually available; keep non-VLM datasets usable.
        has_vlm = vlm.notna()
        out.loc[has_vlm & (decisions != "match"), "final_selected"] = False

    if bool(model_cfg.get("borderline_requires_non_not_match", False)):
        has_vlm = vlm.notna()
        out.loc[has_vlm & (decisions == "not_match"), "final_borderline"] = False

    if stage_d_enabled and bool(sel_cfg.get("require_stage_d_for_selection", False)):
        has_vlm = vlm.notna()
        out.loc[~has_vlm, "final_selected"] = False
        out.loc[~has_vlm, "final_borderline"] = False

    reason = out.get("selection_reason", pd.Series([""] * n, index=idx)).fillna("").astype(str)
    out["selection_reason"] = reason.map(lambda s: s if "generalized_model" in s else (f"{s}|generalized_model" if s else "generalized_model"))
    return out


def _select_final(df: pd.DataFrame, cfg: dict[str, Any], stage_d_enabled: bool) -> pd.DataFrame:
    out = df.copy()
    sel_cfg = cfg.get("selection", {})

    out["final_selected"] = False
    out["final_borderline"] = False
    out["review_score"] = np.nan
    out["selection_reason"] = ""

    stage_d_scores = pd.to_numeric(out.get("stageD_score_0_100", pd.Series([np.nan] * len(out))), errors="coerce")
    if stage_d_enabled and stage_d_scores.notna().any():
        decisions = out.get("stageD_decision", pd.Series([""] * len(out))).fillna("").astype(str)
        out["review_score"] = stage_d_scores.astype(float)
        out["selection_reason"] = "stageD_vlm"
        min_match = int(sel_cfg.get("match_score_min", 80))
        req_decision = str(sel_cfg.get("match_decision_required", "match"))
        out["final_selected"] = (stage_d_scores.fillna(-1).astype(float) >= min_match) & (decisions == req_decision)

        if bool(sel_cfg.get("keep_borderline", True)):
            bmin = int(sel_cfg.get("borderline_score_min", 70))
            out["final_borderline"] = (~out["final_selected"]) & (stage_d_scores.fillna(-1).astype(float) >= bmin)
    elif stage_d_enabled and (
        bool(sel_cfg.get("require_stage_d_for_selection", True))
        and (not bool(sel_cfg.get("allow_non_vlm_fallback_when_stage_d_missing", False)))
    ):
        out["review_score"] = pd.to_numeric(out.get("stageB_clip_score", pd.Series([0.0] * len(out))), errors="coerce").fillna(0.0) * 100.0
        out["selection_reason"] = "stageD_required_no_scores"
        out["final_selected"] = False
        out["final_borderline"] = False
    else:
        # Fallback when Stage D is disabled: use Stage B (+ Stage C if available)
        stage_c_available = out["stageC_pass"].notna().any() if "stageC_pass" in out.columns else False
        stage_b_any_pass = bool(out["stageB_pass"].fillna(False).any()) if "stageB_pass" in out.columns else False
        if stage_c_available and stage_b_any_pass:
            out["review_score"] = out["stageB_clip_score"].fillna(0).astype(float) * 100.0
            out["selection_reason"] = "stageB_plus_stageC"
            out["final_selected"] = out["stageB_pass"].fillna(False) & out["stageC_pass"].fillna(False)
        elif stage_b_any_pass:
            out["review_score"] = out["stageB_clip_score"].fillna(0).astype(float) * 100.0
            out["selection_reason"] = "stageB_only"
            out["final_selected"] = out["stageB_pass"].fillna(False)
        elif "geom_texture_boundary_score" in out.columns and out["geom_texture_boundary_score"].notna().any():
            # Geometry-based fallback when CLIP is unavailable.
            a = out["stageA_rwtd_score"].fillna(0).astype(float)
            g = out["geom_texture_boundary_score"].fillna(0).astype(float)
            obj = out["geom_object_fraction"].fillna(0).astype(float)
            rs = (0.55 * a + 0.45 * g) - 25.0 * obj
            rs = rs.clip(lower=0.0, upper=100.0)
            out["review_score"] = rs
            out["selection_reason"] = "stageA_plus_geometry"
            # Dynamic thresholds for pilot/manual-review mode when no CLIP/VLM exists.
            p90 = float(rs.quantile(0.90))
            p70 = float(rs.quantile(0.70))
            selected_thr = max(30.0, p90)
            borderline_thr = max(24.0, p70)
            out["final_selected"] = rs >= selected_thr
            out["final_borderline"] = (~out["final_selected"]) & (rs >= borderline_thr)
        elif "stageA_rwtd_score" in out.columns and out["stageA_rwtd_score"].notna().any():
            # When Stage B is disabled/unavailable, keep a useful pilot ranking using Stage A quality score.
            s = out["stageA_rwtd_score"].fillna(0).astype(float)
            out["review_score"] = s
            out["selection_reason"] = "stageA_only"
            out["final_selected"] = s >= 75.0
            out["final_borderline"] = (~out["final_selected"]) & (s >= 65.0)
        else:
            out["review_score"] = 0.0
            out["selection_reason"] = "no_signal"
            out["final_selected"] = False

    out = _apply_generalized_selection_model(out, cfg=cfg, stage_d_enabled=stage_d_enabled)
    # Semantic gates must run last so hard rejects cannot be re-selected by downstream scoring.
    out = _apply_semantic_object_gates(out, cfg)

    cap = sel_cfg.get("top_n_cap")
    if cap not in (None, "", 0):
        cap = int(cap)
        selected = out[out["final_selected"]].copy()
        selected = selected.sort_values(
            "stageD_score_0_100" if "stageD_score_0_100" in selected.columns else "stageB_clip_score",
            ascending=False,
        )
        keep_ids = set(selected.head(cap)["image_id"].tolist())
        out["final_selected"] = out["image_id"].isin(keep_ids)

    return out


def _export_selected(df: pd.DataFrame, batch_dir: Path, symlink: bool = True) -> None:
    selected_dir = ensure_dir(batch_dir / "selected")
    borderline_dir = ensure_dir(batch_dir / "borderline")

    for _, row in df[df["final_selected"]].iterrows():
        src = Path(str(row["image_path"]))
        dst = selected_dir / src.name
        if symlink:
            symlink_or_copy(src, dst)
        else:
            dst.write_bytes(src.read_bytes())

    for _, row in df[df["final_borderline"]].iterrows():
        src = Path(str(row["image_path"]))
        dst = borderline_dir / src.name
        if symlink:
            symlink_or_copy(src, dst)
        else:
            dst.write_bytes(src.read_bytes())


def _generate_qa(df: pd.DataFrame, batch_dir: Path, cfg: dict[str, Any]) -> None:
    qa_dir = ensure_dir(batch_dir / "qa")
    qa_cfg = cfg.get("qa", {})
    seed = int(qa_cfg.get("random_seed", 42))
    count = int(qa_cfg.get("random_grid_count", 50))
    top_count = int(qa_cfg.get("top_contact_sheet_count", 100))
    thumb_size = tuple(qa_cfg.get("thumbnail_size", [224, 224]))

    # Top contact sheet.
    rank_col = "stageD_score_0_100" if df["stageD_score_0_100"].notna().any() else "stageB_clip_score"
    top = df.sort_values(rank_col, ascending=False).head(top_count)
    make_contact_sheet(
        [Path(str(p)) for p in top["image_path"].tolist()],
        qa_dir / "contact_sheet_top100.jpg",
        thumb_size=thumb_size,
        title=f"Top {len(top)} images",
    )

    # Stage B pass/fail random samples.
    pass_paths = [Path(str(p)) for p in df[df["stageB_pass"].fillna(False)]["image_path"].tolist()]
    fail_paths = [Path(str(p)) for p in df[~df["stageB_pass"].fillna(False)]["image_path"].tolist()]
    make_contact_sheet(sample_paths(pass_paths, count=count, seed=seed), qa_dir / "stageB_pass_sample_grid.jpg", thumb_size=thumb_size)
    make_contact_sheet(sample_paths(fail_paths, count=count, seed=seed + 1), qa_dir / "stageB_fail_sample_grid.jpg", thumb_size=thumb_size)

    # Histogram.
    score_col = "stageD_score_0_100" if df["stageD_score_0_100"].notna().any() else "stageB_clip_score"
    vals = [float(x) for x in df[score_col].dropna().tolist()]
    score_histogram(vals, qa_dir / "score_histogram.png")

    summary = {
        "n_images": int(len(df)),
        "stageA_pass": int(df["stageA_pass"].fillna(False).sum()) if "stageA_pass" in df.columns else 0,
        "stageB_pass": int(df["stageB_pass"].fillna(False).sum()) if "stageB_pass" in df.columns else 0,
        "stageC_pass": int(df["stageC_pass"].map(lambda x: bool(x) if pd.notna(x) else False).sum())
        if "stageC_pass" in df.columns
        else 0,
        "stageD_scored": int(df["stageD_score_0_100"].notna().sum()) if "stageD_score_0_100" in df.columns else 0,
        "final_selected": int(df["final_selected"].fillna(False).sum()),
        "final_borderline": int(df["final_borderline"].fillna(False).sum()),
    }
    write_json(qa_dir / "summary.json", summary)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>RWTD Batch QA</title>
<style>body{{font-family:Arial,sans-serif;margin:20px;background:#f5f7fa;color:#1f2a37}}
.card{{background:#fff;border:1px solid #d9e1ea;border-radius:10px;padding:14px;max-width:900px}}
code{{background:#eef2f8;padding:2px 6px;border-radius:5px}}</style></head>
<body><div class="card"><h2>RWTD Batch QA</h2>
<p><b>n_images</b>: {summary["n_images"]}</p>
<p><b>stageA_pass</b>: {summary["stageA_pass"]}</p>
<p><b>stageB_pass</b>: {summary["stageB_pass"]}</p>
<p><b>stageC_pass</b>: {summary["stageC_pass"]}</p>
<p><b>stageD_scored</b>: {summary["stageD_scored"]}</p>
<p><b>final_selected</b>: {summary["final_selected"]}</p>
<p><b>final_borderline</b>: {summary["final_borderline"]}</p>
<p>Artifacts: <a href="contact_sheet_top100.jpg">contact_sheet_top100.jpg</a>,
<a href="score_histogram.png">score_histogram.png</a>,
<a href="stageB_pass_sample_grid.jpg">stageB_pass_sample_grid.jpg</a>,
<a href="stageB_fail_sample_grid.jpg">stageB_fail_sample_grid.jpg</a></p>
</div></body></html>"""
    (qa_dir / "report.html").write_text(html, encoding="utf-8")


def _build_balanced_review_subset(df: pd.DataFrame, limit: int, seed: int = 42) -> pd.DataFrame:
    if limit <= 0 or len(df) <= limit:
        return df.copy()

    d = df.copy()
    status = np.where(
        d["final_selected"].fillna(False).astype(bool),
        "selected",
        np.where(d["final_borderline"].fillna(False).astype(bool), "borderline", "rejected"),
    )
    d["__status"] = status

    groups = {
        "selected": d[d["__status"] == "selected"].copy(),
        "borderline": d[d["__status"] == "borderline"].copy(),
        "rejected": d[d["__status"] == "rejected"].copy(),
    }
    ratios = {"selected": 0.35, "borderline": 0.25, "rejected": 0.40}

    alloc: dict[str, int] = {}
    for k, g in groups.items():
        alloc[k] = min(len(g), int(round(limit * ratios[k])))

    allocated = sum(alloc.values())
    if allocated < limit:
        remaining = limit - allocated
        # Fill leftover capacity from larger pools first.
        order = sorted(groups.keys(), key=lambda x: (len(groups[x]) - alloc[x]), reverse=True)
        idx = 0
        while remaining > 0 and order:
            k = order[idx % len(order)]
            cap = len(groups[k]) - alloc[k]
            if cap > 0:
                alloc[k] += 1
                remaining -= 1
            idx += 1
            if idx > 100000:
                break

    picked: list[pd.DataFrame] = []
    for k, g in groups.items():
        take_n = int(alloc.get(k, 0))
        if take_n <= 0 or g.empty:
            continue
        if k == "rejected":
            # Show both hard negatives near threshold and diverse lower-score rejects.
            g_sorted = g.sort_values("review_score", ascending=False)
            top_n = min(max(1, take_n // 2), len(g_sorted))
            top_part = g_sorted.head(top_n)
            rem_n = take_n - len(top_part)
            if rem_n > 0 and len(g_sorted) > top_n:
                rem_pool = g_sorted.iloc[top_n:]
                rem_part = rem_pool.sample(n=min(rem_n, len(rem_pool)), random_state=seed)
                picked.append(pd.concat([top_part, rem_part], ignore_index=True))
            else:
                picked.append(top_part)
        else:
            picked.append(g.sort_values("review_score", ascending=False).head(take_n))

    if not picked:
        return d.sort_values("review_score", ascending=False).head(limit).drop(columns=["__status"], errors="ignore")

    out = pd.concat(picked, ignore_index=True)
    if len(out) > limit:
        out = out.sort_values("review_score", ascending=False).head(limit)
    out = out.drop(columns=["__status"], errors="ignore")
    return out


def _clip_score_to_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    lo = float(s.quantile(0.05))
    hi = float(s.quantile(0.95))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = float(s.min())
        hi = float(s.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        out = pd.Series(np.zeros(len(series), dtype=float), index=series.index)
        out[s.notna()] = 50.0
        return out
    scaled = ((s - lo) / (hi - lo) * 100.0).clip(lower=0.0, upper=100.0)
    return scaled.fillna(0.0)


def _apply_multimodal_fusion_score(df: pd.DataFrame, selected_min: float, borderline_min: float) -> pd.DataFrame:
    out = df.copy()
    # Prefer immutable geometry baseline if present (avoids compounding fusion when rerun).
    base_col = "review_score_geometry_only" if "review_score_geometry_only" in out.columns else "review_score"
    base = pd.to_numeric(out.get(base_col, 0.0), errors="coerce").fillna(0.0).astype(float)
    clip100 = _clip_score_to_100(out.get("stageB_clip_score", pd.Series([np.nan] * len(out))))
    vlm = pd.to_numeric(out.get("stageD_score_0_100", pd.Series([np.nan] * len(out))), errors="coerce")
    vlm_filled = vlm.fillna(0.0).astype(float)

    has_clip = out.get("stageB_clip_score", pd.Series([np.nan] * len(out))).notna()
    has_vlm = vlm.notna()

    # Base fusion: keep geometry as primary signal, CLIP as secondary, VLM as corrective.
    fused = base.copy()
    fused = np.where(has_clip, (0.92 * base + 0.08 * clip100), fused)

    # VLM contributes as a calibration term; it should not dominate the geometric prior.
    fused = np.where(has_vlm, (0.90 * base + 0.08 * clip100 + 0.02 * vlm_filled), fused)

    # Penalize flagged semantic clutter patterns directly from VLM output.
    flags_s = out.get("stageD_flags", pd.Series([""] * len(out))).fillna("").astype(str)
    penalties = np.zeros(len(out), dtype=float)
    penalties += np.where(flags_s.str.contains("object_centric"), 2.0, 0.0)
    penalties += np.where(flags_s.str.contains("too_many_objects"), 1.4, 0.0)
    penalties += np.where(flags_s.str.contains("low_texture_content"), 1.6, 0.0)
    penalties += np.where(flags_s.str.contains("no_clear_texture_boundary"), 1.6, 0.0)
    penalties += np.where(flags_s.str.contains("mosaic_or_collage"), 2.4, 0.0)
    penalties += np.where(flags_s.str.contains("synthetic_or_graphic"), 2.4, 0.0)

    decisions = out.get("stageD_decision", pd.Series([""] * len(out))).fillna("").astype(str)
    fused = fused - penalties
    fused = np.where((has_vlm) & (decisions == "match"), fused + 2.0, fused)
    fused = np.where((has_vlm) & (decisions == "not_match"), fused - 2.0, fused)
    fused = np.clip(fused, 0.0, 100.0)

    out["review_score_base"] = base
    out["review_score"] = fused.astype(float)
    out["selection_reason"] = "multimodal_fusion"
    out["final_selected"] = out["review_score"] >= float(selected_min)
    out["final_borderline"] = (~out["final_selected"]) & (out["review_score"] >= float(borderline_min))
    return out


def _merge_all_batch_manifests(out_root: Path, write_csv: bool = True) -> None:
    manifests = sorted((out_root / "batches").glob("batch_*/batch_manifest.parquet"))
    dfs = []
    for p in manifests:
        try:
            dfs.append(pd.read_parquet(p))
        except Exception:
            csv_p = p.with_suffix(".csv")
            if csv_p.exists():
                dfs.append(pd.read_csv(csv_p))
    if not dfs:
        return
    merged = pd.concat(dfs, ignore_index=True)
    write_table(merged, out_root / "manifests" / "all_processed", write_csv=write_csv)


def _run_one_batch(batch_id: int, batch_df: pd.DataFrame, out_root: Path, cfg: dict[str, Any], skip_vlm: bool) -> pd.DataFrame:
    from rwtd_miner.stages.stage_a_maskstats import (
        apply_stage_a_fallbacks,
        run_stage_a,
        stage_a_pass_rate,
        stage_a_strictness_warning,
    )

    log = get_logger(f"batch_{batch_id}")
    batch_dir = ensure_dir(out_root / "batches" / f"batch_{batch_id:05d}")
    cp = Checkpoint.load(batch_dir / "checkpoint.json")

    manifest_base = batch_dir / "batch_manifest"
    io_cfg = cfg.get("io", {})
    write_csv = bool(io_cfg.get("write_csv_also", True))

    if cp.is_stage_done("complete") and manifest_base.with_suffix(".parquet").exists():
        log.info("Batch %s already complete, loading manifest", batch_id)
        return read_table(manifest_base)

    df = batch_df.copy().reset_index(drop=True)
    df["batch_id"] = int(batch_id)

    runtime_cfg = cfg.get("runtime", {})
    stage_a_cfg = cfg.get("stage_a", {})
    stage_b_cfg = cfg.get("stage_b", {})
    stage_c_cfg = cfg.get("stage_c", {})
    stage_d_cfg = cfg.get("stage_d", {})
    review_cfg = cfg.get("review", {})
    semantic_cfg = cfg.get("semantic_gates", {})

    if stage_a_cfg.get("enabled", True) and not cp.is_stage_done("stage_a"):
        workers = int(runtime_cfg.get("cpu_workers", 8))
        df = run_stage_a(df, cfg=stage_a_cfg, workers=workers)
        df = apply_stage_a_fallbacks(df, stage_a_cfg)
        pass_rate = stage_a_pass_rate(df)
        warn = stage_a_strictness_warning(pass_rate)
        if warn:
            log.warning(warn)
        log.info("Stage A pass rate: %.3f", pass_rate)
        _save_batch_manifest(df, batch_dir, write_csv=write_csv)
        cp.mark_stage_done("stage_a", {"pass_rate": pass_rate})
    elif cp.is_stage_done("stage_a"):
        df = read_table(manifest_base)

    # Geometry + visual explanation stage (mask + texture boundary overlay).
    if bool(review_cfg.get("compute_geometry_debug", True)) and not cp.is_stage_done("stage_geom"):
        df = enrich_with_sa1b_geometry_and_assets(df=df, batch_dir=batch_dir, cfg=review_cfg)
        _save_batch_manifest(df, batch_dir, write_csv=write_csv)
        cp.mark_stage_done("stage_geom")
    elif cp.is_stage_done("stage_geom"):
        df = read_table(manifest_base)

    if bool(semantic_cfg.get("enabled", False)) and not cp.is_stage_done("stage_semantic"):
        df = enrich_with_coco_panoptic_metrics(df=df, cfg=semantic_cfg)
        _save_batch_manifest(df, batch_dir, write_csv=write_csv)
        cp.mark_stage_done("stage_semantic")
    elif cp.is_stage_done("stage_semantic"):
        df = read_table(manifest_base)

    min_short_side = int(cfg.get("index", {}).get("min_short_side", 256))

    if stage_b_cfg.get("enabled", True) and not cp.is_stage_done("stage_b"):
        from rwtd_miner.stages.stage_b_clip import run_stage_b

        df = run_stage_b(df, cfg=stage_b_cfg, runtime_cfg=runtime_cfg, cache_dir=batch_dir / "cache", min_short_side=min_short_side)
        pass_rate_b = float(df["stageB_pass"].fillna(False).mean()) if len(df) else 0.0
        log.info("Stage B pass rate: %.3f", pass_rate_b)
        _save_batch_manifest(df, batch_dir, write_csv=write_csv)
        cp.mark_stage_done("stage_b", {"pass_rate": pass_rate_b})
    elif cp.is_stage_done("stage_b"):
        df = read_table(manifest_base)
    else:
        if "stageB_pos_score" not in df.columns:
            df["stageB_pos_score"] = np.nan
        if "stageB_neg_score" not in df.columns:
            df["stageB_neg_score"] = np.nan
        if "stageB_clip_score" not in df.columns:
            df["stageB_clip_score"] = np.nan
        if "stageB_rank" not in df.columns:
            df["stageB_rank"] = np.nan
        if "stageB_pass" not in df.columns:
            # If Stage B is disabled, allow Stage A-passing rows to continue.
            df["stageB_pass"] = df["stageA_pass"].fillna(False).astype(bool)
        if df["stageB_clip_score"].isna().all() and "stageA_rwtd_score" in df.columns:
            df["stageB_clip_score"] = df["stageA_rwtd_score"].fillna(0).astype(float) / 100.0
        if df["stageB_rank"].isna().all() and "stageB_clip_score" in df.columns:
            order = np.argsort(df["stageB_clip_score"].fillna(-1e9).to_numpy())[::-1]
            ranks = np.empty_like(order, dtype=np.int32)
            ranks[order] = np.arange(1, len(order) + 1)
            df["stageB_rank"] = ranks
        if "stageB_error" not in df.columns:
            df["stageB_error"] = "stage_b_disabled"

    if stage_c_cfg.get("enabled", False) and not cp.is_stage_done("stage_c"):
        from rwtd_miner.stages.stage_c_caption import run_stage_c

        df = run_stage_c(df, cfg=stage_c_cfg, runtime_cfg=runtime_cfg)
        _save_batch_manifest(df, batch_dir, write_csv=write_csv)
        cp.mark_stage_done("stage_c")
    elif cp.is_stage_done("stage_c"):
        df = read_table(manifest_base)
    else:
        if "stageC_caption" not in df.columns:
            df["stageC_caption"] = None
        if "stageC_pass" not in df.columns:
            df["stageC_pass"] = None

    stage_d_enabled = bool(stage_d_cfg.get("enabled", False)) and (not skip_vlm)
    if stage_d_enabled and not cp.is_stage_done("stage_d"):
        from rwtd_miner.stages.stage_d_vlm import run_stage_d

        df = run_stage_d(df, cfg=stage_d_cfg, batch_dir=batch_dir)
        _save_batch_manifest(df, batch_dir, write_csv=write_csv)
        cp.mark_stage_done("stage_d")
    elif cp.is_stage_done("stage_d"):
        df = read_table(manifest_base)
    else:
        if "stageD_score_0_100" not in df.columns:
            df["stageD_score_0_100"] = None
            df["stageD_decision"] = None
            df["stageD_flags"] = None
            df["stageD_reason"] = None
            df["stageD_overlay_score_0_100"] = None
            df["stageD_overlay_pass"] = None
            df["stageD_overlay_reason"] = None

    df = _select_final(df, cfg=cfg, stage_d_enabled=stage_d_enabled)
    _save_batch_manifest(df, batch_dir, write_csv=write_csv)

    _export_selected(df, batch_dir=batch_dir, symlink=bool(io_cfg.get("symlink_selected", True)))
    _generate_qa(df, batch_dir=batch_dir, cfg=cfg)
    review_path = build_review_site(df=df, batch_dir=batch_dir)
    log.info("Review site generated: %s", review_path)

    cp.mark_stage_done("complete", {"n_rows": int(len(df))})

    # Optional cache cleanup for disk pressure.
    if bool(runtime_cfg.get("cleanup_batch_cache", False)):
        cache_dir = batch_dir / "cache"
        if cache_dir.exists():
            for p in cache_dir.glob("*"):
                if p.is_file():
                    p.unlink()

    return df


def cmd_index(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    cfg = load_config(config_path)
    out_root = Path(args.out)
    configure_logging(cfg.get("runtime", {}).get("log_level", "INFO"), log_file=out_root / "logs" / "rwtd_miner.log")

    _build_image_index(
        input_root=Path(args.input_root),
        out_root=out_root,
        cfg=cfg,
        adapter_name=str(args.adapter),
        tfds_name=str(args.tfds_name),
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    cfg = load_config(config_path)

    if args.batch_budget_gb is not None:
        cfg.setdefault("batching", {})["batch_budget_gb"] = float(args.batch_budget_gb)
    if args.max_batches is not None:
        cfg.setdefault("batching", {})["max_batches"] = int(args.max_batches)

    out_root = Path(args.out)
    ensure_dir(out_root)
    configure_logging(cfg.get("runtime", {}).get("log_level", "INFO"), log_file=out_root / "logs" / "rwtd_miner.log")
    log = get_logger("run")

    _, merged = _load_index_or_build(
        input_root=Path(args.input_root),
        out_root=out_root,
        cfg=cfg,
        adapter_name=str(args.adapter),
        tfds_name=str(args.tfds_name),
    )

    max_batches_cfg = cfg.get("batching", {}).get("max_batches")
    max_batches = int(max_batches_cfg) if max_batches_cfg not in (None, "") else None

    unique_batches = sorted(int(x) for x in merged["batch_id"].dropna().unique().tolist())

    if args.batch is not None:
        unique_batches = [int(args.batch)]
    elif max_batches is not None:
        unique_batches = unique_batches[:max_batches]

    if not unique_batches:
        log.warning("No batches to run")
        return 0

    for batch_id in unique_batches:
        batch_df = merged[merged["batch_id"] == batch_id].copy()
        if batch_df.empty:
            continue
        total_bytes = int(batch_df["file_size_bytes"].sum())
        log.info(
            "Running batch %s with %s images (%.2f GB)",
            batch_id,
            len(batch_df),
            total_bytes / (1024**3),
        )
        _run_one_batch(batch_id=batch_id, batch_df=batch_df, out_root=out_root, cfg=cfg, skip_vlm=bool(args.skip_vlm))

    _merge_all_batch_manifests(out_root=out_root, write_csv=bool(cfg.get("io", {}).get("write_csv_also", True)))
    log.info("Run complete. Aggregated manifest: %s", out_root / "manifests" / "all_processed.parquet")
    return 0


def _stage_b_available() -> bool:
    try:
        import importlib.util

        has_torch = importlib.util.find_spec("torch") is not None
        has_open_clip = importlib.util.find_spec("open_clip") is not None
        has_transformers = importlib.util.find_spec("transformers") is not None
        return bool(has_torch and (has_open_clip or has_transformers))
    except Exception:
        return False


def cmd_pilot100(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    cfg = load_config(config_path)
    out_root = Path(args.out)
    ensure_dir(out_root)
    configure_logging(cfg.get("runtime", {}).get("log_level", "INFO"), log_file=out_root / "logs" / "rwtd_miner.log")
    log = get_logger("pilot100")

    pilot_input_root = ensure_dir(out_root / "pilot_input")
    fetch_summary = fetch_sa1b_pairs_from_tar(
        part_url=str(args.part_url),
        out_root=pilot_input_root,
        num_images=int(args.num_images),
        image_target_long_side=int(args.image_target_long_side),
    )
    write_json(out_root / "pilot_fetch_summary.json", fetch_summary)
    log.info("Pilot fetch summary: %s", json.dumps(fetch_summary))

    if not _stage_b_available():
        log.warning("Stage B dependencies not found in current env; pilot will run with Stage B disabled.")
        cfg.setdefault("stage_b", {})["enabled"] = False
        cfg.setdefault("stage_c", {})["enabled"] = False
        cfg.setdefault("stage_d", {})["enabled"] = False

    # Force one pilot batch on the 100-image dataset.
    cfg.setdefault("batching", {})["batch_budget_gb"] = float(max(0.001, args.batch_budget_gb))
    cfg["batching"]["max_batches"] = 1
    cfg.setdefault("runtime", {})["cleanup_batch_cache"] = False

    _build_image_index(input_root=pilot_input_root, out_root=out_root, cfg=cfg, adapter_name="local", tfds_name="segment_anything")
    _, merged = _load_index_or_build(
        input_root=pilot_input_root,
        out_root=out_root,
        cfg=cfg,
        adapter_name="local",
        tfds_name="segment_anything",
    )
    unique_batches = sorted(int(x) for x in merged["batch_id"].dropna().unique().tolist())
    if not unique_batches:
        raise RuntimeError("Pilot index produced no batches")
    batch_id = unique_batches[0]
    batch_df = merged[merged["batch_id"] == batch_id].copy()
    df = _run_one_batch(batch_id=batch_id, batch_df=batch_df, out_root=out_root, cfg=cfg, skip_vlm=bool(args.skip_vlm))

    _merge_all_batch_manifests(out_root=out_root, write_csv=bool(cfg.get("io", {}).get("write_csv_also", True)))
    review_index = out_root / "batches" / f"batch_{batch_id:05d}" / "review" / "index.html"
    print(str(review_index))
    log.info("Pilot complete. Review page: %s", review_index)
    return 0


def _load_manifest_prefer_parquet(base_path: Path) -> pd.DataFrame:
    parquet = base_path.with_suffix(".parquet")
    csv = base_path.with_suffix(".csv")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


def cmd_calibrate_rwtd_reference(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    cfg = load_config(config_path)
    out_root = Path(args.out)
    ensure_dir(out_root)
    configure_logging(cfg.get("runtime", {}).get("log_level", "INFO"), log_file=out_root / "logs" / "rwtd_miner.log")
    log = get_logger("calibrate_rwtd")

    pilot_out = Path(args.pilot_out)
    pilot_batch = pilot_out / "batches" / "batch_00000"
    sa_manifest = _load_manifest_prefer_parquet(pilot_batch / "batch_manifest")
    if sa_manifest.empty:
        raise FileNotFoundError(f"SA-1B pilot manifest not found under: {pilot_batch}")

    # Normalize SA-1B asset links so a combined review page can open them directly.
    sa_review_dir = pilot_batch / "review"
    sa = sa_manifest.copy()
    sa["dataset"] = "sa1b_pilot"
    for col in ("geom_mask_rel", "geom_overlay_rel"):
        if col in sa.columns:
            sa[col] = sa[col].map(
                lambda x: None
                if pd.isna(x) or str(x).strip() == ""
                else str((sa_review_dir / str(x)).resolve()) if not Path(str(x)).is_absolute() else str(Path(str(x)))
            )

    rwtd_root = Path(args.rwtd_root) if args.rwtd_root else (out_root / "reference_datasets" / "Kaust256")
    if not bool(args.skip_download):
        fetch_summary = fetch_rwtd_from_texturesam(rwtd_root, max_images=int(args.max_rwtd_images) if args.max_rwtd_images else None)
        write_json(out_root / "rwtd_fetch_summary.json", fetch_summary)
        log.info("RWTD fetch summary: %s", json.dumps(fetch_summary))
    else:
        log.info("Skipping RWTD download; using existing local folder: %s", rwtd_root)

    combined_batch_dir = ensure_dir(out_root / "batches" / "batch_00000")
    rwtd = score_rwtd_reference_dataset(
        rwtd_root=rwtd_root,
        batch_dir=combined_batch_dir,
        cfg=cfg.get("review", {}),
        max_images=int(args.max_rwtd_images) if args.max_rwtd_images else None,
    )
    if rwtd.empty:
        raise RuntimeError(f"No RWTD rows were created from: {rwtd_root}")

    combined = pd.concat([sa, rwtd], ignore_index=True, sort=False)
    combined = apply_texture_priority_scoring(
        combined,
        selected_min=float(args.selected_min),
        borderline_min=float(args.borderline_min),
    )

    # Persist combined manifest and batch manifest for reproducibility.
    write_csv = bool(cfg.get("io", {}).get("write_csv_also", True))
    write_table(combined, out_root / "manifests" / "combined_sa1b_rwtd", write_csv=write_csv)
    write_table(combined, combined_batch_dir / "batch_manifest", write_csv=write_csv)

    review_index = build_review_site(df=combined, batch_dir=combined_batch_dir)

    summary = {
        "n_total": int(len(combined)),
        "n_sa1b_pilot": int((combined["dataset"] == "sa1b_pilot").sum()),
        "n_rwtd_reference": int((combined["dataset"] == "rwtd_reference").sum()),
        "selected_min": float(args.selected_min),
        "borderline_min": float(args.borderline_min),
        "selected_count": int(combined["final_selected"].fillna(False).sum()),
        "borderline_count": int(combined["final_borderline"].fillna(False).sum()),
        "score_quantiles_by_dataset": {},
    }
    for ds, grp in combined.groupby("dataset", dropna=False):
        key = str(ds)
        vals = grp["review_score"].astype(float)
        summary["score_quantiles_by_dataset"][key] = {
            "count": int(len(grp)),
            "q10": float(vals.quantile(0.10)),
            "q25": float(vals.quantile(0.25)),
            "q50": float(vals.quantile(0.50)),
            "q75": float(vals.quantile(0.75)),
            "q90": float(vals.quantile(0.90)),
            "max": float(vals.max()),
        }
    write_json(out_root / "calibration_summary.json", summary)

    if args.sync_html_to:
        target = Path(args.sync_html_to)
        ensure_dir(target.parent)
        shutil.copytree(review_index.parent, target, dirs_exist_ok=True)
        log.info("Synced review site to: %s", target)

    print(str(review_index))
    log.info("Calibration review site: %s", review_index)
    return 0


def cmd_ade20k_full(args: argparse.Namespace) -> int:
    import pandas as pd
    config_path = Path(args.config)
    cfg = load_config(config_path)
    out_root = Path(args.out)
    ensure_dir(out_root)
    configure_logging(cfg.get("runtime", {}).get("log_level", "INFO"), log_file=out_root / "logs" / "rwtd_miner.log")
    log = get_logger("ade20k_full")

    if args.ade_root:
        ade_root_raw = Path(args.ade_root)
        if (ade_root_raw / "images").exists() and (ade_root_raw / "annotations").exists():
            ade_root = ade_root_raw
        elif (ade_root_raw / "ADEChallengeData2016" / "images").exists():
            ade_root = ade_root_raw / "ADEChallengeData2016"
        else:
            if bool(args.skip_download):
                raise FileNotFoundError(f"Provided --ade_root does not look like ADEChallengeData2016: {ade_root_raw}")
            prep = ensure_ade20k_dataset(ade_root_raw)
            ade_root = Path(str(prep["ade_root"]))
            write_json(out_root / "ade20k_download_summary.json", prep)
    else:
        base = out_root / "reference_datasets"
        if bool(args.skip_download):
            guess = base / "ADEChallengeData2016"
            if not guess.exists():
                raise FileNotFoundError(f"ADE20K not found at {guess}; rerun without --skip_download")
            ade_root = guess
        else:
            prep = ensure_ade20k_dataset(base)
            ade_root = Path(str(prep["ade_root"]))
            write_json(out_root / "ade20k_download_summary.json", prep)

    workers = int(args.workers) if args.workers is not None else int(cfg.get("runtime", {}).get("cpu_workers", 8))
    eval_cfg = {
        "min_short_side": int(cfg.get("index", {}).get("min_short_side", 256)),
        "small_threshold": float(cfg.get("stage_a", {}).get("small_threshold", 0.001)),
        "min_large_region_frac": float(cfg.get("review", {}).get("min_large_region_frac", 0.08)),
        "strong_boundary_min_pixels": int(cfg.get("review", {}).get("strong_boundary_min_pixels", 40)),
    }

    custom_csv = str(getattr(args, "custom_images_csv", "")).strip()
    if custom_csv and Path(custom_csv).exists():
        # --- Custom images mode: Stage A on unique annotations, expand to all variants ---
        log.info("Custom images mode: %s", custom_csv)
        import pandas as pd
        custom_df = pd.read_csv(custom_csv)
        log.info("  Total images in manifest: %d", len(custom_df))

        # Deduplicate by annotation_ref for Stage A (geometry only depends on mask)
        ann_col = "annotation_ref"
        if ann_col in custom_df.columns:
            unique_anns = custom_df.drop_duplicates(subset=[ann_col]).copy()
            log.info("  Unique annotations for Stage A: %d", len(unique_anns))

            # Build pairs for run_ade20k_full_eval-compatible format
            pairs = []
            for _, row in unique_anns.iterrows():
                ann_path = str(row[ann_col])
                # Use annotation stem as image_id for Stage A
                ann_stem = Path(ann_path).stem
                # Find any image_path for this annotation (just for Stage A metadata)
                img_path = str(row.get("image_path", ""))
                pairs.append({
                    "image_id": f"training_{ann_stem}" if not ann_stem.startswith("training_") else ann_stem,
                    "image_path": img_path,
                    "annotation_ref": ann_path,
                })

            # Run Stage A on unique annotations
            from rwtd_miner.utils.ade20k_reference import (
                build_ade20k_class_map, _analyze_one, apply_texture_priority_scoring,
            )
            class_map = build_ade20k_class_map(ade_root)
            from concurrent.futures import ProcessPoolExecutor, as_completed
            from tqdm import tqdm

            stageA_rows = []
            with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
                futs = [ex.submit(_analyze_one, row, class_map, eval_cfg) for row in pairs]
                for f in tqdm(as_completed(futs), total=len(futs), desc="stageA_geometry", unit="img"):
                    stageA_rows.append(f.result())

            stageA_df = pd.DataFrame(stageA_rows)
            stageA_df = apply_texture_priority_scoring(
                stageA_df,
                selected_min=float(args.selected_min),
                borderline_min=float(args.borderline_min),
            )

            # Vectorized merge: join Stage A scores onto all variants via annotation_ref
            score_cols = [c for c in stageA_df.columns if c not in ("image_id", "image_path")]
            stageA_lookup = stageA_df[score_cols].copy()
            stageA_lookup[ann_col] = stageA_df["annotation_ref"].astype(str)
            stageA_lookup = stageA_lookup.drop_duplicates(subset=[ann_col])

            df = custom_df.copy()
            df[ann_col] = df[ann_col].astype(str)
            df = df.merge(stageA_lookup, on=ann_col, how="left")

            # Ensure required columns exist
            if "dataset" not in df.columns:
                df["dataset"] = "custom"
            if "review_score" not in df.columns or df["review_score"].isna().all():
                df["review_score"] = 0.0
            df["review_score"] = df["review_score"].fillna(0.0)
            df = df.sort_values("review_score", ascending=False).reset_index(drop=True)

            log.info("  Expanded DataFrame: %d images with Stage A scores", len(df))
            log.info("  review_score range: %.1f - %.1f (mean %.1f)",
                      df["review_score"].min(), df["review_score"].max(), df["review_score"].mean())
        else:
            # No annotations — skip Stage A, just load images
            df = custom_df.copy()
            df["review_score"] = 50.0  # neutral score
            df["final_selected"] = True
            df["final_borderline"] = False

        summary = {
            "n_total": len(df),
            "selected_min": float(args.selected_min),
            "borderline_min": float(args.borderline_min),
            "selected_count": int(df.get("final_selected", pd.Series(dtype=bool)).fillna(False).sum()),
            "borderline_count": int(df.get("final_borderline", pd.Series(dtype=bool)).fillna(False).sum()),
            "rejected_count": int(len(df) - df.get("final_selected", pd.Series(dtype=bool)).fillna(False).sum()
                                  - df.get("final_borderline", pd.Series(dtype=bool)).fillna(False).sum()),
            "selected_ratio": 0.0,
        }
        baseline_summary = dict(summary)
    else:
        df, summary = run_ade20k_full_eval(
            ade_root=ade_root,
            out_root=out_root,
            cfg=eval_cfg,
            selected_min=float(args.selected_min),
            borderline_min=float(args.borderline_min),
            workers=workers,
        )
        baseline_summary = dict(summary)

    if bool(args.enable_clip):
        from rwtd_miner.stages.stage_b_clip import run_stage_b

        log.info("Running Stage B (CLIP) over ADE20K candidates")
        df = run_stage_b(
            df=df,
            cfg=cfg.get("stage_b", {}),
            runtime_cfg=cfg.get("runtime", {}),
            cache_dir=ensure_dir(out_root / "cache_clip"),
            min_short_side=int(cfg.get("index", {}).get("min_short_side", 256)),
        )

    if bool(args.enable_vlm):
        from rwtd_miner.stages.stage_d_vlm import run_stage_d

        stage_d_cfg = dict(cfg.get("stage_d", {}))
        stage_d_cfg["backend"] = str(args.vlm_backend)
        if str(args.vlm_backend) == "external_command" and str(args.vlm_external_command).strip():
            stage_d_cfg["external_command"] = str(args.vlm_external_command).strip()
        stage_d_cfg["score_top_n_from_stage_b"] = int(args.vlm_top_n)
        stage_d_cfg["device_preference"] = str(args.vlm_device)
        stage_d_cfg["hf_blip_vqa_model_name"] = str(args.vlm_model_name)
        log.info(
            "Running Stage D (VLM) backend=%s top_n=%s model=%s",
            stage_d_cfg["backend"],
            stage_d_cfg["score_top_n_from_stage_b"],
            stage_d_cfg["hf_blip_vqa_model_name"],
        )
        df = run_stage_d(df=df, cfg=stage_d_cfg, batch_dir=out_root)

    if bool(args.enable_clip) or bool(args.enable_vlm):
        df = _apply_multimodal_fusion_score(
            df=df,
            selected_min=float(args.selected_min),
            borderline_min=float(args.borderline_min),
        )
        df = df.sort_values("review_score", ascending=False).reset_index(drop=True)
        total = int(len(df))
        selected = int(df["final_selected"].fillna(False).sum())
        borderline = int(df["final_borderline"].fillna(False).sum())
        rejected = int(total - selected - borderline)
        summary = {
            "n_total": total,
            "selected_min": float(args.selected_min),
            "borderline_min": float(args.borderline_min),
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
        multimodal_report = {
            "baseline_geometry_only": baseline_summary,
            "multimodal": summary,
            "clip_scored_count": int(pd.to_numeric(df.get("stageB_clip_score"), errors="coerce").notna().sum()),
            "clip_pass_count": int(df.get("stageB_pass", pd.Series(dtype=bool)).fillna(False).sum()),
            "vlm_scored_count": int(pd.to_numeric(df.get("stageD_score_0_100"), errors="coerce").notna().sum()),
            "vlm_decision_counts": {
                "match": int((df.get("stageD_decision", pd.Series(dtype=str)) == "match").sum()),
                "borderline": int((df.get("stageD_decision", pd.Series(dtype=str)) == "borderline").sum()),
                "not_match": int((df.get("stageD_decision", pd.Series(dtype=str)) == "not_match").sum()),
            },
        }
        write_json(out_root / "ade20k_eval_summary_multimodal.json", multimodal_report)

    write_csv = bool(cfg.get("io", {}).get("write_csv_also", True))
    write_table(df, out_root / "manifests" / "ade20k_all", write_csv=write_csv)

    selected_dir = ensure_dir(out_root / "selected_minbar")
    for _, row in df[df["final_selected"].fillna(False)].iterrows():
        src = Path(str(row["image_path"]))
        ext = src.suffix.lower() or ".jpg"
        dst = selected_dir / f"{str(row['image_id'])}{ext}"
        symlink_or_copy(src, dst)

    review_limit = int(args.review_limit)
    # Skip review site for custom CSV mode (too many images) or when review_limit=0
    custom_csv_active = bool(str(getattr(args, "custom_images_csv", "")).strip()) and Path(str(getattr(args, "custom_images_csv", ""))).exists()
    if not custom_csv_active:
        batch_dir = ensure_dir(out_root / "batches" / "batch_00000")
        if review_limit > 0 and len(df) > review_limit:
            review_df = _build_balanced_review_subset(df=df, limit=review_limit, seed=int(cfg.get("seed", 42)))
        else:
            review_df = df.copy()
        review_df = render_ade20k_review_assets(
            df=review_df,
            batch_dir=batch_dir,
            class_map=build_ade20k_class_map(ade_root),
            cfg=eval_cfg,
            max_items=review_limit if review_limit > 0 else None,
        )
        write_table(review_df, batch_dir / "batch_manifest", write_csv=write_csv)
        review_index = build_review_site(df=review_df, batch_dir=batch_dir)
    else:
        log.info("Skipping review site generation (custom images mode, %d images)", len(df))
        review_index = None

    if args.sync_html_to and review_index is not None:
        target = Path(args.sync_html_to)
        ensure_dir(target.parent)
        shutil.copytree(review_index.parent, target, dirs_exist_ok=True)
        log.info("Synced review site to: %s", target)

    log.info(
        "ADE20K complete: total=%s selected=%s borderline=%s rejected=%s",
        summary["n_total"],
        summary["selected_count"],
        summary["borderline_count"],
        summary["rejected_count"],
    )
    print(str(review_index))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detecture")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Build image index + batch assignments")
    p_index.add_argument("--input_root", required=True)
    p_index.add_argument("--out", required=True)
    p_index.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    p_index.add_argument("--adapter", default="local", choices=["local", "tfds"])
    p_index.add_argument("--tfds_name", default="segment_anything")
    p_index.set_defaults(func=cmd_index)

    p_run = sub.add_parser("run", help="Run mining pipeline on batches")
    p_run.add_argument("--input_root", required=True)
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    p_run.add_argument("--adapter", default="local", choices=["local", "tfds"])
    p_run.add_argument("--tfds_name", default="segment_anything")
    p_run.add_argument("--batch", type=int, default=None)
    p_run.add_argument("--batch_budget_gb", type=float, default=None)
    p_run.add_argument("--max_batches", type=int, default=None)
    p_run.add_argument("--skip_vlm", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_pilot = sub.add_parser("pilot100", help="Fetch 100 SA-1B pairs and build review site")
    p_pilot.add_argument("--out", required=True)
    p_pilot.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    p_pilot.add_argument("--num_images", type=int, default=100)
    p_pilot.add_argument("--batch_budget_gb", type=float, default=0.6)
    p_pilot.add_argument("--image_target_long_side", type=int, default=1024)
    p_pilot.add_argument(
        "--part_url",
        default="https://huggingface.co/datasets/Aber-r/SA-1B_backup/resolve/main/sa_000000.tar",
    )
    p_pilot.add_argument("--skip_vlm", action="store_true")
    p_pilot.set_defaults(func=cmd_pilot100)

    p_cal = sub.add_parser("calibrate_rwtd", help="Combine SA-1B pilot with RWTD reference and build calibrated review site")
    p_cal.add_argument("--pilot_out", required=True)
    p_cal.add_argument("--out", required=True)
    p_cal.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    p_cal.add_argument("--rwtd_root", default="")
    p_cal.add_argument("--max_rwtd_images", type=int, default=256)
    p_cal.add_argument("--selected_min", type=float, default=60.0)
    p_cal.add_argument("--borderline_min", type=float, default=50.0)
    p_cal.add_argument("--skip_download", action="store_true")
    p_cal.add_argument("--sync_html_to", default="")
    p_cal.set_defaults(func=cmd_calibrate_rwtd_reference)

    p_ade = sub.add_parser("ade20k_full", help="Download/evaluate ADE20K and count images above RWTD-calibrated bar")
    p_ade.add_argument("--out", required=True)
    p_ade.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    p_ade.add_argument("--ade_root", default="")
    p_ade.add_argument("--skip_download", action="store_true")
    p_ade.add_argument("--workers", type=int, default=None)
    p_ade.add_argument("--selected_min", type=float, default=60.0)
    p_ade.add_argument("--borderline_min", type=float, default=50.0)
    p_ade.add_argument("--review_limit", type=int, default=1500)
    p_ade.add_argument("--enable_clip", action="store_true")
    p_ade.add_argument("--enable_vlm", action="store_true")
    p_ade.add_argument(
        "--vlm_backend",
        default="hf_blip_vqa",
        choices=["hf_blip_vqa", "hf_vlm_chat", "external_command", "stub"],
    )
    p_ade.add_argument("--vlm_external_command", default="")
    p_ade.add_argument("--vlm_model_name", default="Salesforce/blip-vqa-base")
    p_ade.add_argument("--vlm_device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p_ade.add_argument("--vlm_top_n", type=int, default=1200)
    p_ade.add_argument("--sync_html_to", default="")
    p_ade.add_argument("--custom_images_csv", default="",
                       help="CSV with columns: image_id,image_path,annotation_ref. "
                            "Stage A runs on unique annotations, then CLIP+VLM run on all images.")
    p_ade.set_defaults(func=cmd_ade20k_full)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
