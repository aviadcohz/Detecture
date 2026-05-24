#!/usr/bin/env python3
"""
Pipeline orchestrator: runs all steps sequentially.

Usage:
    # Full pipeline (score + extract + describe + validate)
    python scripts/run_pipeline.py --dataset ade20k --steps score extract describe validate

    # Just hit Run in IDE (default: extract → describe → validate, scoring assumed done)
    python scripts/run_pipeline.py

    # Test run: extract 5 images, describe 5, skip scoring
    python scripts/run_pipeline.py --extract-limit 5 --describe-limit 5
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, get_output_dir


STEPS = {
    "score": "01_score.py",
    "extract": "02_extract.py",
    "describe": "03_describe.py",
    "validate": "04_validate.py",
    "expand_degrees": "05_expand_degrees.py",
}

DEFAULT_ORDER = ["extract", "describe", "validate"]
ALL_STEPS = ["score", "extract", "describe", "validate"]
ALL_STEPS_EXPANDED = ["score", "extract", "expand_degrees", "describe", "validate"]


def print_summary(config: dict):
    """Print per-stage funnel showing how many images survive each step."""
    output_dir = get_output_dir(config)
    scoring_dir = output_dir / "scoring"
    metadata_path = output_dir / config["output"]["metadata_file"]
    images_dir = output_dir / config["output"]["image_dir"]
    threshold = config["extract"]["score_threshold"]

    print(f"\n{'='*60}")
    print("  PIPELINE FUNNEL")
    print(f"{'='*60}\n")

    # Stage 1: Scored
    n_scored = 0
    n_above_threshold = 0
    parquets = list(scoring_dir.glob("**/*.parquet")) if scoring_dir.exists() else []
    if parquets:
        import pandas as pd
        df = pd.read_parquet(parquets[0])
        n_scored = len(df)
        n_above_threshold = int((df["review_score"] >= threshold).sum())

    # Stage 2: Extracted
    n_extracted = len(list(images_dir.iterdir())) if images_dir.exists() else 0

    # Stage 3+4: Described & validated
    n_described = 0
    n_textures = 0
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        n_described = len(metadata)
        n_textures = sum(len(e.get("textures", [])) for e in metadata)

    print(f"  Score:    {n_scored:>6} images scored")
    print(f"  Extract:  {n_above_threshold:>6} passed threshold (score >= {threshold})")
    print(f"            {n_extracted:>6} images with masks extracted")
    print(f"  Describe: {n_described:>6} images with Claude descriptions")
    print(f"  Output:   {n_textures:>6} total texture regions")
    print(f"\n  Output dir: {output_dir}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run data preparation pipeline")
    parser.add_argument("--dataset", default="ade20k", help="Dataset name (default: ade20k)")
    parser.add_argument("--steps", nargs="+", choices=STEPS.keys(), default=None,
                        help="Which steps to run (default: extract describe validate)")
    parser.add_argument("--all", action="store_true",
                        help="Run ALL steps including scoring")
    parser.add_argument("--all-expanded", action="store_true",
                        help="Run ALL steps + expand_degrees + re-describe")
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to each step")
    parser.add_argument("--describe-limit", type=int, default=None,
                        help="Limit samples for describe step (for testing)")
    parser.add_argument("--extract-limit", type=int, default=None,
                        help="Limit samples for extract step")
    parser.add_argument("--test-load", action="store_true", default=True,
                        help="Test DetectureDataset loading in validate step")
    args = parser.parse_args()

    if args.steps:
        steps = args.steps
    elif args.all_expanded:
        steps = ALL_STEPS_EXPANDED
    elif args.all:
        steps = ALL_STEPS
    else:
        steps = DEFAULT_ORDER

    scripts_dir = Path(__file__).parent
    project_root = scripts_dir.parent

    # Setup logging to file + stdout
    from utils import get_output_dir
    log_config = load_config(
        str(project_root / "configs" / "base.yaml"),
        str(project_root / "configs" / f"{args.dataset}.yaml"),
    )
    log_dir = get_output_dir(log_config) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"pipeline_{timestamp}.log"
    log_file = open(log_path, "w", buffering=1)  # line-buffered

    import io

    def log(msg):
        """Print to stdout and write to log file."""
        print(msg, flush=True)
        log_file.write(msg + "\n")

    log(f"\n{'='*60}")
    log(f"  Detecture Training Data Pipeline")
    log(f"  Dataset: {args.dataset}")
    log(f"  Steps: {' → '.join(steps)}")
    log(f"  Log: {log_path}")
    log(f"{'='*60}")

    for step in steps:
        script = scripts_dir / STEPS[step]
        log(f"\n{'='*60}")
        log(f"  STEP: {step} ({script.name})")
        log(f"{'='*60}\n")

        cmd = [sys.executable, str(script), "--dataset", args.dataset]

        if args.dry_run and step in ("score", "describe"):
            cmd.append("--dry-run")

        if step == "extract" and args.extract_limit:
            cmd.extend(["--limit", str(args.extract_limit)])

        if step == "describe" and args.describe_limit:
            cmd.extend(["--limit", str(args.describe_limit)])

        if step == "validate" and args.test_load:
            cmd.append("--test-load")

        # Run subprocess, tee output to both stdout and log file
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            line = line.rstrip("\n")
            print(line, flush=True)
            log_file.write(line + "\n")
        process.wait()

        if process.returncode != 0:
            log(f"\nERROR: Step '{step}' failed with return code {process.returncode}")
            log_file.close()
            sys.exit(1)

    # Load config for summary
    config = log_config
    print_summary(config)

    log(f"{'='*60}")
    log(f"  PIPELINE COMPLETE")
    log(f"  Log saved: {log_path}")
    log(f"{'='*60}\n")
    log_file.close()


if __name__ == "__main__":
    main()
