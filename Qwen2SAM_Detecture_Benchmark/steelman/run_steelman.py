#!/usr/bin/env python3
"""Steelman benchmark driver — runs all 3 models × {RWTD, STLD} and prints
a side-by-side summary table of mIoU / ARI.

Each (model × dataset) cell is executed as a fresh subprocess so GPU
memory is released between runs (the Qwen model loads ~16 GB in bf16
and cannot co-reside with a second SAM3 forward-graph).

Usage:
    python run_steelman.py                       # all 3 × {RWTD, STLD}
    python run_steelman.py --model qwen          # one model, both datasets
    python run_steelman.py --dataset RWTD        # all models, one dataset
    python run_steelman.py --limit 10            # smoke test

Per-cell JSON lands in `results/<model>/<dataset>/zero_shot_results.json`;
the aggregate table is written to `results/steelman_summary.json` +
printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


_STEELMAN_ROOT = Path(__file__).resolve().parent
_RESULTS_ROOT = _STEELMAN_ROOT / "results"


# name → (description, script filename)
MODELS = {
    "qwen": ("Qwen2SAM-DeTexture (autonomous)",       "eval_qwen2sam_e2e.py"),
    "gsam3": ("Grounded-SAM3 (top-1 box + inverse)",  "eval_gsam3_top1_inverse.py"),
    "sam3":  ("SAM3 vanilla ('texture' + inverse)",   "eval_sam3_vanilla_inverse.py"),
}

DATASETS = ["RWTD", "STLD"]


def run_one(model_key: str, dataset: str, limit: int | None,
            dry_run: bool) -> dict:
    desc, script = MODELS[model_key]
    out_dir = _RESULTS_ROOT / model_key / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(_STEELMAN_ROOT / script),
        "--dataset", dataset,
        "--output-dir", str(out_dir),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    header = f"[{model_key:>6s} × {dataset:<5s}]"
    print(f"\n{header} {desc}")
    print(f"{header} $ {' '.join(cmd)}", flush=True)
    if dry_run:
        return {"status": "dry_run", "cmd": cmd, "output_dir": str(out_dir)}

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    status = "ok" if proc.returncode == 0 else "failed"
    print(f"{header} done rc={proc.returncode} ({elapsed:.1f}s)", flush=True)
    return {
        "status": status, "returncode": proc.returncode,
        "elapsed_seconds": elapsed, "output_dir": str(out_dir),
    }


def load_cell_summary(out_dir: Path) -> dict | None:
    path = out_dir / "zero_shot_results.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())["summary"]
    except Exception:                                           # noqa: BLE001
        return None


def print_table(rows: list[dict]):
    print("\n" + "=" * 90)
    print("  STEELMAN BENCHMARK — mIoU / ARI side-by-side")
    print("=" * 90)
    print(f"  {'Model':<38s} {'Dataset':<6s} "
          f"{'pIoU':>7s} {'mIoU':>7s} {'ARI':>7s} {'n':>5s} {'extras':<20s}")
    print("  " + "-" * 88)
    for r in rows:
        extras = []
        if "mean_k_pred" in r:
            extras.append(f"<k>={r['mean_k_pred']:.2f}")
        if "n_no_detection" in r:
            extras.append(f"no-det={r['n_no_detection']}")
        extras_s = " ".join(extras)
        pIoU = r.get("panoptic_iou")
        mIoU = r.get("matched_mean_iou")
        ARI = r.get("mean_ari")
        n_ok = r.get("n_ok", 0)
        print(f"  {r['_label']:<38s} {r['_dataset']:<6s} "
              f"{pIoU:>7.4f} {mIoU:>7.4f} {ARI:>7.4f} {n_ok:>5d} {extras_s:<20s}"
              if pIoU is not None
              else f"  {r['_label']:<38s} {r['_dataset']:<6s} "
                   f"{'—':>7s} {'—':>7s} {'—':>7s} {n_ok:>5d} {extras_s:<20s}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", default=None,
                        choices=list(MODELS.keys()),
                        help="Run only these models (default: all 3).")
    parser.add_argument("--dataset", nargs="+", default=None,
                        choices=DATASETS,
                        help="Run only these datasets (default: RWTD, STLD).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Sample cap per cell (debug).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-out", type=Path,
                        default=_RESULTS_ROOT / "steelman_summary.json")
    args = parser.parse_args()

    models = args.model or list(MODELS.keys())
    datasets = args.dataset or DATASETS

    print(f"[runner] models:   {models}")
    print(f"[runner] datasets: {datasets}")
    print(f"[runner] results:  {_RESULTS_ROOT}")

    runs = []
    for m in models:
        for d in datasets:
            res = run_one(m, d, args.limit, args.dry_run)
            runs.append({"model": m, "dataset": d, **res})

    # Collect per-cell summaries for the table.
    rows = []
    for run in runs:
        cell = load_cell_summary(Path(run["output_dir"]))
        row = {
            "_label": MODELS[run["model"]][0],
            "_dataset": run["dataset"],
            "runner_status": run["status"],
        }
        if cell is not None:
            row.update(cell)
        rows.append(row)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(
        {"runs": runs, "cells": rows}, indent=2, default=str,
    ))

    if not args.dry_run:
        print_table(rows)
    print(f"\n  summary JSON: {args.summary_out}")


if __name__ == "__main__":
    main()
