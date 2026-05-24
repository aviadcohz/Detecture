#!/usr/bin/env python3
"""
Step 2: Filter by score and extract images + texture masks from raw dataset.

For ADE20K: reads scoring parquet, filters by review_score (or picked list),
then for each selected image creates:
  - Per-class binary masks (all instances of same class merged)
  - A single index mask (annotation visualization)
  - A colored overlay for debugging

Output: extract_manifest.json + images/ + masks/ + overlays/
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, get_output_dir, ensure_dirs
from adapters.ade20k import (
    ADE20KAdapter, extract_class_masks, create_index_mask,
    create_colored_visualization, create_overlay,
)
from adapters.ade20k_texturesam import ADE20KTextureSAMAdapter
from adapters.ADE20K_textured_images import ADE20KTextureSAMFullAdapter

ADAPTERS = {
    "ade20k": ADE20KAdapter,
    "ade20k_texturesam": ADE20KTextureSAMAdapter,
    "ADE20K_textured_images": ADE20KTextureSAMFullAdapter,
}


def main():
    parser = argparse.ArgumentParser(description="Extract filtered data from raw dataset")
    parser.add_argument("--dataset", required=True, choices=ADAPTERS.keys())
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of extracted samples")
    parser.add_argument("--min-regions", type=int, default=2,
                        help="Minimum texture regions required per image (default: 2)")
    parser.add_argument("--max-regions", type=int, default=5,
                        help="Maximum texture regions per image (default: 5)")
    parser.add_argument("--min-region-frac", type=float, default=0.01,
                        help="Minimum fraction of image for a region (default: 0.01 = 1%%)")
    parser.add_argument("--no-score-filter", action="store_true",
                        help="Skip score filtering, scan raw images directly")
    parser.add_argument("--picked-list", type=str, default=None,
                        help="Path to text file with picked image IDs (overrides score filter)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config = load_config(
        str(project_root / "configs" / "base.yaml"),
        str(project_root / "configs" / f"{args.dataset}.yaml"),
    )

    adapter = ADAPTERS[args.dataset](config)
    output_dir = get_output_dir(config)
    dirs = ensure_dirs(output_dir, config)

    score_threshold = config["extract"]["score_threshold"]

    # --- Determine which images to process ---
    if args.picked_list:
        # Use explicit picked list
        with open(args.picked_list) as f:
            picked_ids = set(
                line.strip().replace("ade20k/", "")
                for line in f if line.strip()
            )
        print(f"Using picked list: {len(picked_ids)} images from {args.picked_list}")

        # Load scoring to get image paths
        scored = adapter.load_scored_images(dirs["scoring"])
        scored_map = {e["image_id"]: e for e in scored}

        image_entries = []
        for pid in picked_ids:
            if pid in scored_map:
                image_entries.append(scored_map[pid])
            else:
                # Construct entry from ID
                image_entries.append({
                    "image_id": pid,
                    "review_score": 0.0,
                    "image_path": "",
                    "annotation_ref": "",
                })
        image_entries.sort(key=lambda x: x.get("review_score", 0), reverse=True)

    elif args.no_score_filter:
        if hasattr(adapter, "scan_images"):
            # Adapter provides its own image scanning (e.g., TextureSAM)
            image_entries = adapter.scan_images()
        else:
            ade_root = Path(config["source"]["ade_root"])
            images_dir = ade_root / "images" / "training"
            image_entries = []
            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    image_entries.append({
                        "image_id": f"training_{img_path.stem}",
                        "review_score": 100.0,
                        "image_path": str(img_path),
                        "annotation_ref": "",
                    })
        print(f"No-score-filter mode: {len(image_entries)} images")

    else:
        scored = adapter.load_scored_images(dirs["scoring"])
        if not scored:
            print("ERROR: No scoring results found. Run 01_score.py first.")
            sys.exit(1)
        print(f"Loaded {len(scored)} scored images")
        image_entries = [e for e in scored if e["review_score"] >= score_threshold]
        image_entries.sort(key=lambda x: x["review_score"], reverse=True)
        print(f"Images with review_score >= {score_threshold}: {len(image_entries)}")

    # --- Extract images, masks, overlays ---
    manifest = []
    skipped_no_masks = 0
    skipped_missing = 0
    seen_ids = set()

    for entry in image_entries:
        if args.limit and len(manifest) >= args.limit:
            break

        image_id = adapter.get_entry_id(entry)
        if image_id in seen_ids:
            continue
        seen_ids.add(image_id)

        image_path = adapter.get_image_path(entry)
        annotation_path = adapter.get_annotation_path(entry)

        if not Path(image_path).exists():
            print(f"  Missing image: {image_path}")
            skipped_missing += 1
            continue
        if not Path(annotation_path).exists():
            print(f"  Missing annotation: {annotation_path}")
            skipped_missing += 1
            continue

        # Extract per-class masks (all instances merged per class)
        region_data = extract_class_masks(
            annotation_path,
            adapter.class_names,
            min_region_frac=args.min_region_frac,
            max_regions=args.max_regions,
        )

        if len(region_data) < args.min_regions:
            skipped_no_masks += 1
            continue

        # Read image
        image_bgr = cv2.imread(image_path)
        h, w = image_bgr.shape[:2]

        # Resize masks if annotation has different size than image
        # (e.g., TextureSAM textured images are resized from originals)
        resized_region_data = []
        for mask, cid, cname in region_data:
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            resized_region_data.append((mask, cid, cname))
        region_data = resized_region_data

        # Save image
        dst_image = dirs["images"] / f"{image_id}.jpg"
        shutil.copy2(image_path, dst_image)

        # Save per-class binary masks
        dst_mask_paths = []
        class_names_list = []
        masks_list = []
        for i, (mask, cid, cname) in enumerate(region_data):
            mask_name = f"{image_id}_mask_{i}.png"
            dst_mask = dirs["textures_mask"] / mask_name
            cv2.imwrite(str(dst_mask), mask)
            dst_mask_paths.append(str(dst_mask))
            class_names_list.append(cname)
            masks_list.append(mask)

        # Create index mask and save as annotation visualization
        index_mask = create_index_mask(masks_list, h, w)
        ann_vis = create_colored_visualization(index_mask)
        dst_ann_vis = dirs["masks"] / f"{image_id}_annotation.png"
        cv2.imwrite(str(dst_ann_vis), ann_vis)

        # Create and save overlay
        overlay_img = create_overlay(image_bgr, index_mask)
        dst_overlay = dirs["overlays"] / f"{image_id}.jpg"
        cv2.imwrite(str(dst_overlay), overlay_img)

        manifest.append({
            "id": image_id,
            "source_image_id": image_id,
            "review_score": entry.get("review_score", 0.0),
            "image_path": str(dst_image),
            "mask_paths": dst_mask_paths,
            "annotation_mask_path": str(dst_ann_vis),
            "overlay_path": str(dst_overlay),
            "existing_descriptions": [],  # Claude will generate these
            "class_names": class_names_list,
            "num_textures": len(region_data),
        })

        if len(manifest) % 50 == 0:
            print(f"  Extracted {len(manifest)} images...")

    # Save manifest
    manifest_path = output_dir / "extract_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_masks = sum(e["num_textures"] for e in manifest)
    print(f"\nExtraction complete:")
    print(f"  Extracted: {len(manifest)} images ({total_masks} texture masks)")
    print(f"  Avg regions/image: {total_masks / max(1, len(manifest)):.1f}")
    print(f"  Skipped (< {args.min_regions} regions): {skipped_no_masks}")
    print(f"  Skipped (missing files): {skipped_missing}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
