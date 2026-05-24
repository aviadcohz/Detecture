#!/usr/bin/env python3
"""Paper benchmark runner — single entry point a reviewer can execute to
reproduce every number in the results tables.

Evaluates four models on up to four datasets under a strict, dataset-
dependent regime. The (model, dataset) pair determines which underlying
eval script is spawned and with which flags:

  Datasets → K-regime
    CAID              : k1          (single region per image)
    RWTD, STLD        : k2          (oracle K=2; inverse-trick for baselines)
    ADE20k_Detecture  : autonomous  (K hidden; no model sees K_GT)

  Models
    detecture         (ours) trained Qwen2SAM_Detecture, best.pt
    sam3_vanilla      SAM3 text encoder → semantic head
    grounded_sam3     Grounding DINO → SAM3 box-prompt
    sa2va             ByteDance Sa2VA-4B end-to-end VLM

Per-regime dispatch (see DISPATCH below) ensures every model is evaluated
under the matching fairness constraint: on RWTD/STLD every model gets the
K=2 prior (via prompt, [m1,-m1] inverse, or explicit top-K); on ADE20k
nobody gets K_GT (no-k-peek flags active); on CAID K=1 is trivial and the
default scripts suffice.

Each cell is spawned as a subprocess so GPU state is clean per model
(Detecture ~16 GB in bf16 cannot co-reside with a second backbone).

Typical usage:

    # Reproduce everything the paper reports (all 4 models × all datasets)
    python master_runner.py

    # Restrict to one dataset (e.g. quick smoke on RWTD)
    python master_runner.py --dataset RWTD

    # One model across all datasets
    python master_runner.py --model detecture

    # One cell, 10-sample smoke
    python master_runner.py --model sa2va --dataset STLD --limit 10

    # Only print the commands — don't spawn
    python master_runner.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml


SUITE_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = SUITE_ROOT / "datasets_config.yaml"
RESULTS_ROOT = SUITE_ROOT / "results"


# --------------------------------------------------------------------- #
# Regime + dispatch tables                                                #
# --------------------------------------------------------------------- #

REGIME_BY_DATASET = {
    "CAID":             "k1",
    "RWTD":             "k2",
    "STLD":             "k2",
    "ADE20k_Detecture": "autonomous",
}

# (model, regime) → (script_path_relative_to_SUITE_ROOT, extra_cli_args)
# Scripts listed here all accept: --dataset, --output-dir, --limit.
DISPATCH: dict[tuple[str, str], tuple[str, list[str]]] = {
    # ---- Detecture (ours): E2E live generation, prompt-level K ------- #
    ("detecture", "k1"):
        ("steelman/eval_qwen2sam_e2e.py", ["--prompt-n", "1"]),
    ("detecture", "k2"):
        ("steelman/eval_qwen2sam_e2e.py", ["--prompt-n", "2"]),
    ("detecture", "autonomous"):
        ("steelman/eval_qwen2sam_e2e.py", ["--prompt-n", "between 1 and 6"]),

    # ---- SAM3 vanilla ------------------------------------------------ #
    # k1 : static mode, single-text "water surface" (config default).
    # k2 : inverse-trick (text="texture" → [m1,-m1]) — same K=2 advantage
    #      every other baseline gets, expressed mathematically.
    # aut: proposal-decoder in its *native* top-K_GT mode. SAM3's mask-
    #      proposal decoder emits a confidence-ranked mask set with no
    #      prior over how many to keep; running it blind with a static
    #      confidence threshold filters everything to dustbin (the model's
    #      scores for generic "texture regions" are structurally below 0.5
    #      even on correct masks). Taking top-K_GT is the only way to
    #      elicit a usable output from this baseline on ADE20K — same
    #      concession the TextureSAM paper makes.
    ("sam3_vanilla", "k1"):
        ("eval_sam3_vanilla.py", []),
    ("sam3_vanilla", "k2"):
        ("steelman/eval_sam3_vanilla_inverse.py", []),
    ("sam3_vanilla", "autonomous"):
        ("eval_sam3_vanilla.py", []),

    # ---- Grounded-SAM3 (DINO → SAM3 box-prompt → semantic head) ------- #
    # k1 : min(n_det, K_GT) collapses to top-1 box — benign on K=1 data.
    # k2 : steelman top-1 box → [m1,-m1].
    # aut: top-N by DINO score above threshold, max_n=6.
    ("grounded_sam3", "k1"):
        ("eval_grounded_sam3.py", []),
    ("grounded_sam3", "k2"):
        ("steelman/eval_gsam3_top1_inverse.py", []),
    ("grounded_sam3", "autonomous"):
        ("eval_grounded_sam3.py", ["--no-k-peek", "--max-n", "6"]),

    # ---- Sa2VA (end-to-end VLM with interleaved [SEG]) --------------- #
    # k1/k2: K=2 prompt from config + code-level top-K truncation.
    # aut  : no-k-peek, cap at max_n=6; Sa2VA emits whatever it wants.
    ("sa2va", "k1"):
        ("eval_vlm_end2end.py", ["--backend", "sa2va"]),
    ("sa2va", "k2"):
        ("eval_vlm_end2end.py", ["--backend", "sa2va"]),
    ("sa2va", "autonomous"):
        ("eval_vlm_end2end.py", ["--backend", "sa2va",
                                 "--no-k-peek", "--max-n", "6"]),
}

# Paper-facing display names.
DISPLAY_NAMES = {
    "detecture":     "Detecture (ours)",
    "sam3_vanilla":  "SAM3",
    "grounded_sam3": "Grounded-SAM3",
    "sa2va":         "Sa2VA",
}

# One-line regime descriptors printed in the header of the summary table.
REGIME_LABELS = {
    "k1":         "K=1 oracle",
    "k2":         "K=2 oracle (inverse trick for baselines)",
    "autonomous": "blind — no K_GT leak",
}


# --------------------------------------------------------------------- #
# Config + argument resolution                                            #
# --------------------------------------------------------------------- #

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_targets(cfg: dict, model: list[str] | None,
                    dataset: list[str] | None):
    known_models = [m["name"] for m in cfg["models"]]
    known_datasets = list(cfg["datasets"].keys())

    models = model or known_models
    datasets = dataset or known_datasets

    bad_m = [m for m in models if m not in known_models]
    if bad_m:
        raise SystemExit(f"unknown --model {bad_m}. Known: {known_models}")
    bad_d = [d for d in datasets if d not in known_datasets]
    if bad_d:
        raise SystemExit(f"unknown --dataset {bad_d}. Known: {known_datasets}")

    # Enforce that every (model, dataset-regime) pair we're about to run
    # has an entry in DISPATCH. Fail loudly if not — missing entries are
    # how "cheating K" silently re-enters the benchmark.
    missing = []
    for m in models:
        for d in datasets:
            regime = REGIME_BY_DATASET.get(d)
            if regime is None:
                raise SystemExit(
                    f"dataset {d!r} is not mapped to a regime in "
                    f"REGIME_BY_DATASET; add it or skip it via --dataset."
                )
            if (m, regime) not in DISPATCH:
                missing.append((m, d, regime))
    if missing:
        lines = "\n".join(
            f"  - {m!s:<16s} × {d!s:<18s}  (regime={r})"
            for m, d, r in missing
        )
        raise SystemExit(
            f"DISPATCH has no entry for these (model, regime) pairs:\n{lines}\n"
            "Add them (or restrict --model / --dataset) before running."
        )

    return models, datasets


def build_cmd(model: str, dataset: str, limit: int | None,
              no_vis: bool) -> tuple[Path, list[str]]:
    regime = REGIME_BY_DATASET[dataset]
    script_rel, extra = DISPATCH[(model, regime)]
    out_dir = RESULTS_ROOT / model / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(SUITE_ROOT / script_rel),
        "--dataset", dataset,
        "--output-dir", str(out_dir),
    ]
    cmd += list(extra)
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if no_vis and "--no-vis" not in cmd:
        # Only add --no-vis for scripts that accept it (the legacy
        # benchmark scripts). The steelman scripts don't produce vis.
        legacy_scripts = {
            "eval_sam3_vanilla.py",
            "eval_grounded_sam3.py",
            "eval_vlm_end2end.py",
        }
        if Path(script_rel).name in legacy_scripts:
            cmd += ["--no-vis"]
    return out_dir, cmd


# --------------------------------------------------------------------- #
# Per-cell execution                                                      #
# --------------------------------------------------------------------- #

def run_one(model: str, dataset: str, limit: int | None,
            no_vis: bool, dry_run: bool) -> dict:
    regime = REGIME_BY_DATASET[dataset]
    out_dir, cmd = build_cmd(model, dataset, limit, no_vis)

    header = f"[{model:>14s} × {dataset:<18s} regime={regime}]"
    print(f"\n{header} {DISPLAY_NAMES[model]} on {dataset}")
    print(f"{header} $ {shlex.join(cmd)}", flush=True)
    if dry_run:
        return {"status": "dry_run", "cmd": cmd,
                "output_dir": str(out_dir), "regime": regime}

    if not Path(cmd[1]).exists():
        print(f"{header} SKIP — eval script not found: {cmd[1]}", flush=True)
        return {"status": "missing_script",
                "output_dir": str(out_dir), "regime": regime}

    t0 = time.time()
    proc = subprocess.run(cmd)
    elapsed = time.time() - t0
    status = "ok" if proc.returncode == 0 else "failed"
    print(f"{header} done rc={proc.returncode} ({elapsed:.1f}s) → {out_dir}",
          flush=True)
    return {
        "status": status, "returncode": proc.returncode,
        "elapsed_seconds": elapsed, "output_dir": str(out_dir),
        "regime": regime,
    }


# --------------------------------------------------------------------- #
# Per-cell summary loading + final table                                  #
# --------------------------------------------------------------------- #

def load_cell_summary(out_dir: Path) -> dict | None:
    path = out_dir / "zero_shot_results.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("summary")
    except Exception:                                           # noqa: BLE001
        return None


def print_final_table(rows: list[dict]):
    col_model = max(len(r["_label"]) for r in rows) + 2 if rows else 20
    print("\n" + "=" * 100)
    print("  PAPER BENCHMARK — mIoU / ARI summary (4 models × 4 datasets)")
    print("=" * 100)
    print(f"  {'Model':<{col_model}s} "
          f"{'Dataset':<18s} {'Regime':<11s} "
          f"{'pIoU':>7s} {'mIoU':>7s} {'ARI':>7s} {'n':>5s}")
    print("  " + "-" * 98)
    for r in rows:
        pIoU = r.get("panoptic_iou")
        mIoU = r.get("matched_mean_iou")
        ARI = r.get("mean_ari")
        n_ok = r.get("n_ok", 0)
        if pIoU is None:
            metrics_s = f"{'—':>7s} {'—':>7s} {'—':>7s}"
        else:
            metrics_s = f"{pIoU:>7.4f} {mIoU:>7.4f} {ARI:>7.4f}"
        print(f"  {r['_label']:<{col_model}s} "
              f"{r['_dataset']:<18s} {r['_regime']:<11s} "
              f"{metrics_s} {n_ok:>5d}")
    print("=" * 100)


# --------------------------------------------------------------------- #
# Main                                                                    #
# --------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--model", type=str, nargs="+", default=None,
                        help="Run only these models (default: all 4). "
                             "Known: detecture, sam3_vanilla, grounded_sam3, sa2va.")
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        help="Run only these datasets (default: every dataset "
                             "in config).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap samples per cell (debug smoke tests).")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip PNG visualizations in legacy scripts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print dispatched commands without executing.")
    parser.add_argument("--summary-out", type=Path,
                        default=RESULTS_ROOT / "paper_benchmark_summary.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    models, datasets = resolve_targets(cfg, args.model, args.dataset)

    print(f"[runner] models:   {models}")
    print(f"[runner] datasets: {datasets} "
          f"(regimes: {[REGIME_BY_DATASET[d] for d in datasets]})")
    print(f"[runner] results:  {RESULTS_ROOT}")
    print(f"[runner] regimes:  {REGIME_LABELS}")

    runs = []
    for m in models:
        for d in datasets:
            res = run_one(m, d, args.limit, args.no_vis, args.dry_run)
            runs.append({"model": m, "dataset": d, **res})

    # Collect per-cell summaries → table rows.
    rows = []
    for run in runs:
        cell = load_cell_summary(Path(run["output_dir"]))
        row = {
            "_label": DISPLAY_NAMES[run["model"]],
            "_model": run["model"],
            "_dataset": run["dataset"],
            "_regime": run["regime"],
            "runner_status": run["status"],
        }
        if cell is not None:
            row.update(cell)
        rows.append(row)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(
        {
            "regime_labels": REGIME_LABELS,
            "regime_by_dataset": REGIME_BY_DATASET,
            "runs": runs, "cells": rows,
        },
        indent=2, default=str,
    ))

    overall = {
        "n_runs": len(runs),
        "n_ok": sum(1 for r in runs if r["status"] == "ok"),
        "n_failed": sum(1 for r in runs if r["status"] == "failed"),
        "n_missing": sum(1 for r in runs if r["status"] == "missing_script"),
        "n_dry_run": sum(1 for r in runs if r["status"] == "dry_run"),
    }

    if not args.dry_run:
        print_final_table(rows)

    print(f"\n  {overall['n_ok']} ok   "
          f"{overall['n_failed']} failed   "
          f"{overall['n_missing']} missing-script   "
          f"(of {overall['n_runs']} cells)")
    print(f"  summary JSON: {args.summary_out}")
    if not args.dry_run:
        print(f"  → for paper-ready CSV / LaTeX tables, run:")
        print(f"      python aggregate_results.py --csv results/summary.csv "
              f"--latex results/summary.tex")


if __name__ == "__main__":
    main()
