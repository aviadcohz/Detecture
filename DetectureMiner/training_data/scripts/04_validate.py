#!/usr/bin/env python3
"""
Step 4: Validate output and convert to final Qwen2SAM_Detecture metadata format.

Reads describe_manifest.json, validates all entries, and produces the final
metadata.json compatible with DetectureDataset.

Output: metadata.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, get_output_dir


def validate_image(path: str) -> tuple[bool, str]:
    """Check image loads as RGB."""
    img = cv2.imread(path)
    if img is None:
        return False, f"Cannot load image: {path}"
    if len(img.shape) != 3 or img.shape[2] != 3:
        return False, f"Image is not RGB: {path} shape={img.shape}"
    return True, ""


def validate_mask_file(path: str) -> tuple[bool, str]:
    """Check mask is grayscale binary (0/255)."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False, f"Cannot load mask: {path}"
    unique_vals = set(np.unique(mask))
    if not unique_vals.issubset({0, 255}):
        return False, f"Non-binary mask: {path} values={unique_vals}"
    if 255 not in unique_vals:
        return False, f"Empty mask (no foreground): {path}"
    return True, ""


def validate_description(desc: str, prefix: str = "Texture of ") -> tuple[bool, str]:
    """Check description format."""
    if not desc.startswith(prefix):
        return False, f"Missing prefix '{prefix}': {desc}"
    # Count words after prefix
    words = desc[len(prefix):].split()
    if len(words) < 5:
        return False, f"Too short ({len(words)} words): {desc}"
    if len(words) > 25:
        return False, f"Too long ({len(words)} words): {desc}"
    return True, ""


def convert_to_qwen2sam_format(sample: dict) -> dict:
    """Convert a describe_manifest entry to Qwen2SAM_Detecture metadata format."""
    mask_paths = sample["mask_paths"]
    descriptions = sample["descriptions"]

    # descriptions is a dict like {"texture_a": "...", "texture_b": "..."}
    textures = []
    for i, mp in enumerate(mask_paths):
        key = f"texture_{chr(ord('a') + i)}"
        desc = descriptions.get(key, "")
        textures.append({"description": desc, "mask_path": mp})

    return {
        "image_path": sample["image_path"],
        "textures": textures,
        "source_image_id": sample.get("source_image_id", ""),
        "id": sample.get("id", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate and convert to final format")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--test-load", action="store_true",
                        help="Test-load with DetectureDataset")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config = load_config(
        str(project_root / "configs" / "base.yaml"),
        str(project_root / "configs" / f"{args.dataset}.yaml"),
    )
    output_dir = get_output_dir(config)

    # Load describe manifest
    describe_path = output_dir / "describe_manifest.json"
    if not describe_path.exists():
        print("ERROR: describe_manifest.json not found. Run 03_describe.py first.")
        sys.exit(1)

    with open(describe_path) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} described samples")

    # Validate and convert
    metadata = []
    errors = []

    for sample in samples:
        sample_id = sample["id"]
        sample_errors = []

        # Validate image
        if config["validate"]["check_images"]:
            ok, err = validate_image(sample["image_path"])
            if not ok:
                sample_errors.append(err)

        # Validate masks
        if config["validate"]["check_masks"]:
            for mp in sample["mask_paths"]:
                ok, err = validate_mask_file(mp)
                if not ok:
                    sample_errors.append(err)

        # Validate descriptions
        if config["validate"]["check_descriptions"]:
            descriptions = sample.get("descriptions", {})
            n_masks = len(sample.get("mask_paths", []))
            for i in range(n_masks):
                key = f"texture_{chr(ord('a') + i)}"
                desc = descriptions.get(key, "")
                ok, err = validate_description(desc)
                if not ok:
                    sample_errors.append(err)

        if sample_errors:
            errors.append({"id": sample_id, "errors": sample_errors})
            print(f"  FAIL {sample_id}: {sample_errors[0]}")
            continue

        entry = convert_to_qwen2sam_format(sample)
        metadata.append(entry)

    # Save final metadata
    metadata_path = output_dir / config["output"]["metadata_file"]
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save validation errors
    if errors:
        errors_path = output_dir / "validation_errors.json"
        with open(errors_path, "w") as f:
            json.dump(errors, f, indent=2)

    print(f"\nValidation complete:")
    print(f"  Valid: {len(metadata)}")
    print(f"  Invalid: {len(errors)}")
    print(f"  Output: {metadata_path}")

    # Test-load with DetectureDataset
    if args.test_load and config["validate"]["test_dataset_load"]:
        print("\nTesting DetectureDataset compatibility...")
        try:
            dataset_py = config["validate"]["qwen2sam_dataset_path"]
            # Import DetectureDataset dynamically
            import importlib.util
            spec = importlib.util.spec_from_file_location("dataset", dataset_py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            ds = mod.DetectureDataset(str(metadata_path), image_size=256)
            print(f"  Dataset loaded: {len(ds)} samples")

            # Try loading first sample
            sample = ds[0]
            print(f"  Sample keys: {list(sample.keys())}")
            print(f"  SAM image shape: {sample['sam_image'].shape}")
            print(f"  Index mask shape: {sample['index_mask'].shape}")
            print(f"  Index mask unique: {sample['index_mask'].unique().tolist()}")
            print(f"  k_gt: {sample['k_gt']}")
            print(f"  Descriptions: {sample['descriptions']}")
            print("  DetectureDataset test PASSED")
        except Exception as e:
            print(f"  DetectureDataset test FAILED: {e}")


if __name__ == "__main__":
    main()
