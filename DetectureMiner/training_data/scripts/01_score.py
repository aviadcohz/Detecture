#!/usr/bin/env python3
"""
Step 1: Run Detecture scoring pipeline.

Invokes the Detecture CLI to compute review_score (0-100) for each image.
Output: parquet/csv manifest in output/<dataset>/scoring/
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, get_output_dir, ensure_dirs
from adapters import ADE20KAdapter
from adapters.ade20k_texturesam import ADE20KTextureSAMAdapter
from adapters.ADE20K_textured_images import ADE20KTextureSAMFullAdapter

ADAPTERS = {
    "ade20k": ADE20KAdapter,
    "ade20k_texturesam": ADE20KTextureSAMAdapter,
    "ADE20K_textured_images": ADE20KTextureSAMFullAdapter,
}


def main():
    parser = argparse.ArgumentParser(description="Run Detecture scoring")
    parser.add_argument("--dataset", required=True, choices=ADAPTERS.keys())
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config = load_config(
        str(project_root / "configs" / "base.yaml"),
        str(project_root / "configs" / f"{args.dataset}.yaml"),
    )

    adapter = ADAPTERS[args.dataset](config)
    output_dir = get_output_dir(config)
    dirs = ensure_dirs(output_dir, config)

    # Update output path in command to use absolute path
    cmd = adapter.get_score_command()
    # Replace relative output with absolute
    for i, arg in enumerate(cmd):
        if arg == str(Path(config["output"]["base_dir"]) / config["dataset"] / "scoring"):
            cmd[i] = str(dirs["scoring"])

    detecture_root = config["score"]["detecture_root"]

    print(f"Detecture scoring for {args.dataset}")
    print(f"  Working dir: {detecture_root}")
    print(f"  Output: {dirs['scoring']}")
    print(f"  Command: {' '.join(cmd)}")

    if args.dry_run:
        print("\n[DRY RUN] Command not executed.")
        return

    # Check for existing results
    parquet_files = list(dirs["scoring"].glob("**/*.parquet"))
    if parquet_files:
        print(f"\nFound existing scoring results: {parquet_files[0]}")
        print("To re-run, delete the scoring directory first.")
        return

    # Ensure conda env lib path is available (fixes libstdc++ issues)
    import os
    env = os.environ.copy()
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        conda_lib = str(Path(conda_prefix) / "lib")
        env["LD_LIBRARY_PATH"] = conda_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(
        cmd,
        cwd=detecture_root,
        capture_output=False,
        env=env,
    )

    if result.returncode != 0:
        print(f"\nScoring failed with return code {result.returncode}")
        sys.exit(1)

    # Verify output
    parquet_files = list(dirs["scoring"].glob("**/*.parquet"))
    if parquet_files:
        import pandas as pd
        df = pd.read_parquet(parquet_files[0])
        print(f"\nScoring complete: {len(df)} images scored")
        print(f"  review_score range: {df['review_score'].min():.1f} - {df['review_score'].max():.1f}")
        print(f"  >= 80: {(df['review_score'] >= 80).sum()} images")
    else:
        print("\nWarning: No parquet output found after scoring.")


if __name__ == "__main__":
    main()
