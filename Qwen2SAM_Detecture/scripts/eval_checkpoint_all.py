#!/usr/bin/env python3
"""Run BOTH Oracle (teacher-forced) and E2E (live exactly-2) evaluation on
one or many checkpoints, writing results in the format
`regenerate_unified_plots.py` consumes.

Usage — single checkpoint:
    python scripts/eval_checkpoint_all.py --checkpoint checkpoints/epoch_12.pt

Usage — batch (run sequentially, one after the other):
    python scripts/eval_checkpoint_all.py \\
        --checkpoint checkpoints/epoch_4.pt \\
                     checkpoints/epoch_8.pt \\
                     checkpoints/epoch_12.pt \\
                     checkpoints/epoch_16.pt

    # or with a shell glob:
    python scripts/eval_checkpoint_all.py --checkpoint checkpoints/epoch_*.pt

For each checkpoint, writes up to two JSONs under `checkpoints/test_results/`:
    <stem>_bridge_oracle.json       — Oracle (teacher-forced); `mean_iou`
    <stem>_e2e_exactly_2.json       — E2E live, exactly-2 prompt; `mean_iou`
                                       (extracted from ablation summary.json
                                       condition_B_exactly_2 branch)

E2E is auto-skipped for checkpoints before epoch 12 (configurable via
`--e2e-min-epoch`), since those predate the Masked-row/Shifted-Zero
SEG-emission fix and return 0.0 by construction — only Oracle is meaningful.

Wall-time per checkpoint: Oracle ~1.5 min + E2E ~25 min.
Running ep4, 8, 12, 16 → 2×(Oracle) + 2×(Oracle+E2E) ≈ 56 min.

After the batch, refresh plots:
    python scripts/regenerate_unified_plots.py
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list, cwd: Path = None) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=cwd)


def eval_one_checkpoint(ckpt: Path, args, test_dir: Path) -> int:
    """Evaluate a single checkpoint. Returns 0 on success, non-zero on error."""
    stem = ckpt.stem
    epoch = _infer_epoch_from_stem(stem)

    # ---- Oracle ---- #
    if not args.skip_oracle:
        oracle_out = test_dir / f"{stem}_bridge_oracle.json"
        if oracle_out.exists() and not args.force:
            print(f"  [skip] {oracle_out.name} already exists (use --force to redo)")
        else:
            print(f"\n  --- ORACLE (teacher-forced) → {oracle_out.name} ---")
            rc = run(["python", "scripts/evaluate_bridge_oracle.py",
                      "--checkpoint", str(ckpt),
                      "--config", args.config,
                      "--output-json", str(oracle_out)],
                     cwd=_PROJECT_ROOT)
            if rc != 0:
                return rc

    # ---- E2E live exactly-2 ---- #
    # Pre-ep12 checkpoints predate the Masked-row/Shifted-Zero SEG-emission fix,
    # so live E2E returns 0.0 by construction — only Oracle is meaningful.
    skip_e2e_for_epoch = epoch > 0 and epoch < args.e2e_min_epoch
    if skip_e2e_for_epoch and not args.skip_e2e:
        print(f"  [skip E2E] epoch {epoch} < {args.e2e_min_epoch} "
              f"(pre-SEG-emission-fix; Oracle only)")

    if not args.skip_e2e and not skip_e2e_for_epoch:
        ablation_dir = test_dir / f"{stem}_live_ablation"
        flat_path = test_dir / f"{stem}_e2e_exactly_2.json"
        if flat_path.exists() and not args.force:
            print(f"  [skip] {flat_path.name} already exists (use --force to redo)")
        else:
            print(f"\n  --- E2E LIVE EXACTLY-2 → {ablation_dir.name}/ ---")
            rc = run(["python", "scripts/ablation_exact_k2_rwtd.py",
                      "--checkpoint", str(ckpt),
                      "--config", args.config,
                      "--output-dir", str(ablation_dir)],
                     cwd=_PROJECT_ROOT)
            if rc != 0:
                return rc

            # Flatten the ablation output into a single consumer-friendly file.
            summary_path = ablation_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                body = summary.get("condition_B_exactly_2", {})
                flat = {
                    "checkpoint": str(ckpt),
                    "displayed_epoch": _infer_epoch_from_stem(stem),
                    "n_samples": summary.get("n_samples"),
                    "mean_iou": body.get("mean_iou"),
                    "mean_ari": body.get("mean_ari"),
                    "k_dist": body.get("k_dist"),
                    "condition_A_1_to_6": summary.get("condition_A_1_to_6"),
                    "condition_B_exactly_2": body,
                    "note": "E2E live-inference, honest k=2 constraint. "
                            "Source: ablation_exact_k2_rwtd.py summary.json.",
                }
                with open(flat_path, "w") as f:
                    json.dump(flat, f, indent=2)
                print(f"  Flattened E2E summary → {flat_path.name}")

    return 0


def main():
    p = argparse.ArgumentParser(
        description="Run Oracle + E2E eval on one or many checkpoints.")
    p.add_argument("--checkpoint", required=True, nargs="+",
                   help="One or more checkpoint paths. "
                        "Shell globs like checkpoints/epoch_*.pt are supported.")
    p.add_argument("--config", default="configs/detecture.yaml")
    p.add_argument("--test-results-dir",
                   default=str(_PROJECT_ROOT / "checkpoints/test_results"))
    p.add_argument("--skip-oracle", action="store_true")
    p.add_argument("--skip-e2e", action="store_true")
    p.add_argument("--e2e-min-epoch", type=int, default=12,
                   help="Only run E2E live eval for checkpoints at/after this "
                        "epoch. Pre-ep12 checkpoints predate the "
                        "Masked-row/Shifted-Zero SEG-emission fix, so live "
                        "inference returns 0.0 by construction. Default: 12.")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if output JSONs already exist.")
    args = p.parse_args()

    # Resolve + validate all checkpoint paths up front so we fail fast.
    ckpts = []
    for c in args.checkpoint:
        p_ = Path(c).resolve()
        if not p_.exists():
            print(f"ERROR: {p_} not found.")
            sys.exit(1)
        ckpts.append(p_)

    test_dir = Path(args.test_results_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"Will evaluate {len(ckpts)} checkpoint(s):")
    n_e2e = 0
    for c in ckpts:
        ep = _infer_epoch_from_stem(c.stem)
        will_e2e = (not args.skip_e2e) and (ep == 0 or ep >= args.e2e_min_epoch)
        tag = "Oracle+E2E" if will_e2e else "Oracle only"
        print(f"  - {c.name}  [{tag}]  (epoch={ep or '?'})")
        if will_e2e:
            n_e2e += 1
    print(f"Output dir: {test_dir}")
    est_min = len(ckpts) * (0 if args.skip_oracle else 1.5) \
              + n_e2e * 25
    print(f"Estimated total wall time: ~{est_min:.0f} min")

    for i, ckpt in enumerate(ckpts, start=1):
        print(f"\n{'='*66}")
        print(f"  [{i}/{len(ckpts)}]  {ckpt.name}")
        print(f"{'='*66}")
        rc = eval_one_checkpoint(ckpt, args, test_dir)
        if rc != 0:
            print(f"\nAborting: checkpoint {ckpt} failed (rc={rc}). "
                  f"Fix then re-run — completed checkpoints won't rerun "
                  f"unless you pass --force.")
            sys.exit(rc)

    print(f"\n{'='*66}")
    print(f"All {len(ckpts)} checkpoint(s) evaluated.")
    print(f"Regenerate plots with:")
    print(f"    python scripts/regenerate_unified_plots.py")
    print(f"{'='*66}")


def _infer_epoch_from_stem(stem: str) -> int:
    """Best-effort extraction of a displayed-epoch number from a checkpoint
    filename like 'epoch_12', 'ep10_snapshot', 'best'."""
    import re
    m = re.search(r"epoch[_\s]*(\d+)", stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"ep(\d+)", stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0  # caller should override if needed


if __name__ == "__main__":
    main()
