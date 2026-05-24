#!/usr/bin/env python3
"""
Workaround: Add degree=1.0 images with score >= 30 to the dataset.

Standalone script — does NOT modify the pipeline. Adds images directly
to the output directory with masks, overlays, and empty descriptions.

Usage:
    python scripts/workaround_add_degree1.py
    python scripts/workaround_add_degree1.py --dry-run
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from adapters.ade20k import (
    _parse_object_info, extract_class_masks,
    create_index_mask, create_overlay,
)

OUTPUT_DIR = Path.home() / "detecture_training_data" / "output" / "ADE20K_textured_images"
SCORING_DIR = OUTPUT_DIR / "scoring" / "manifests"
IMAGES_DIR = OUTPUT_DIR / "images"
MASKS_DIR = OUTPUT_DIR / "textures_mask"
OVERLAYS_DIR = OUTPUT_DIR / "overlays"
METADATA_PATH = OUTPUT_DIR / "metadata.json"
TEXTURED_DIR = Path.home() / "ADE20K_textured_images" / "images_flat"
ADE_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "ade20k" / "ADEChallengeData2016"
ANNOTATIONS_DIR = ADE_ROOT / "annotations" / "training"

# Separate review folder for visual inspection
REVIEW_DIR = OUTPUT_DIR / "degree1_review"

SCORE_THRESHOLD = 30
MIN_REGIONS = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load class names
    class_names = _parse_object_info(ADE_ROOT / "objectInfo150.txt")

    # Load scoring
    parquets = list(SCORING_DIR.glob("*.parquet"))
    if not parquets:
        print("ERROR: No scoring parquet found")
        sys.exit(1)
    df = pd.read_parquet(parquets[0])

    # Load existing metadata
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
    else:
        print(f"ERROR: {METADATA_PATH} not found. Run pipeline first.")
        sys.exit(1)

    existing_ids = set(e.get("source_image_id", e.get("id", "")) for e in metadata)
    print(f"Current dataset: {len(metadata)} images")

    # Find degree=1.0 with score >= 30, not already in dataset
    degree1 = df[
        df["image_id"].str.contains("degree1.0")
        & (df["review_score"] >= SCORE_THRESHOLD)
        & (~df["image_id"].isin(existing_ids))
    ]
    print(f"Degree=1.0, score >= {SCORE_THRESHOLD}, not in dataset: {len(degree1)}")

    if len(degree1) == 0:
        print("Nothing to add.")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would add {len(degree1)} images")
        return

    # Create review folder for visual inspection
    review_images = REVIEW_DIR / "images"
    review_overlays = REVIEW_DIR / "overlays"
    review_images.mkdir(parents=True, exist_ok=True)
    review_overlays.mkdir(parents=True, exist_ok=True)
    print(f"\nReview folder: {REVIEW_DIR}")

    # Process each image
    added = 0
    skipped = 0
    for _, row in degree1.iterrows():
        image_id = row["image_id"]
        base_id = image_id.replace("_textured_degree1.0", "")

        # Source image
        src_image = TEXTURED_DIR / f"{image_id}.jpg"
        if not src_image.exists():
            skipped += 1
            continue

        # Annotation
        ann_path = ANNOTATIONS_DIR / f"{base_id}.png"
        if not ann_path.exists():
            skipped += 1
            continue

        # Extract masks
        regions = extract_class_masks(str(ann_path), class_names,
                                      min_region_frac=0.01, max_regions=10)
        if len(regions) < MIN_REGIONS:
            skipped += 1
            continue

        # Copy image
        dst_image = IMAGES_DIR / f"{image_id}.jpg"
        if not dst_image.exists():
            shutil.copy2(str(src_image), str(dst_image))

        # Save masks (resized to image resolution)
        image_bgr_tmp = cv2.imread(str(src_image))
        img_h, img_w = image_bgr_tmp.shape[:2] if image_bgr_tmp is not None else (512, 512)
        mask_paths = []
        names = []
        for i, (mask, cid, name) in enumerate(regions):
            if mask.shape[:2] != (img_h, img_w):
                mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            mask_path = MASKS_DIR / f"{image_id}_mask_{i}.png"
            if not mask_path.exists():
                cv2.imwrite(str(mask_path), mask)
            mask_paths.append(str(mask_path))
            names.append(name)

        # Create overlay
        image_bgr = cv2.imread(str(dst_image))
        if image_bgr is not None:
            h, w = image_bgr.shape[:2]
            # Resize masks to image resolution (annotations may differ)
            resized_masks = []
            for mask, _, _ in regions:
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                resized_masks.append(mask)
            idx_mask = create_index_mask(resized_masks, h, w)
            overlay = create_overlay(image_bgr, idx_mask)
            overlay_path = OVERLAYS_DIR / f"{image_id}_overlay.jpg"
            if not overlay_path.exists():
                cv2.imwrite(str(overlay_path), overlay)

            # Copy to review folder for inspection
            shutil.copy2(str(dst_image), str(review_images / f"{image_id}.jpg"))
            cv2.imwrite(str(review_overlays / f"{image_id}_overlay.jpg"), overlay)
        else:
            overlay_path = ""

        # Build metadata entry
        textures = []
        for i, (mask_path, name) in enumerate(zip(mask_paths, names)):
            textures.append({
                "description": "",  # filled by Claude retry
                "mask_path": mask_path,
                "class_name": name,
            })

        metadata.append({
            "image_path": str(dst_image),
            "textures": textures,
            "source_image_id": image_id,
            "id": image_id,
            "review_score": float(row["review_score"]),
            "overlay_path": str(overlay_path),
            "num_textures": len(textures),
            "degree": 1.0,
            "added_by": "workaround_degree1",
        })
        added += 1

        if added % 100 == 0:
            print(f"  Processed {added}...")

    # Save updated metadata
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save review summary
    review_summary = {
        "added": added,
        "skipped": skipped,
        "score_threshold": SCORE_THRESHOLD,
        "min_regions": MIN_REGIONS,
        "images_dir": str(review_images),
        "overlays_dir": str(review_overlays),
    }
    with open(REVIEW_DIR / "summary.json", "w") as f:
        json.dump(review_summary, f, indent=2)

    print(f"\nDone!")
    print(f"  Added: {added}")
    print(f"  Skipped: {skipped}")
    print(f"  New total: {len(metadata)} images")
    print(f"  Entries needing descriptions: {sum(1 for e in metadata for t in e['textures'] if not t.get('description', '').strip())}")
    print(f"\n  Review the added images at:")
    print(f"    Images:   {review_images}")
    print(f"    Overlays: {review_overlays}")


if __name__ == "__main__":
    main()
