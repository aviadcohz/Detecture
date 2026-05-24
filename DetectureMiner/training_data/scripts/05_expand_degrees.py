#!/usr/bin/env python3
"""
Step 5: Expand dataset with higher-degree TextureSAM variants.

Reads extract_manifest.json (output of 02_extract.py). For each image
that passed at degree D, adds all variants with degree > D.
Masks are reused from the base annotation. Produces metadata.json with
empty descriptions (filled by 03_describe.py).

Usage:
    python scripts/05_expand_degrees.py --dataset ADE20K_textured_images
    python scripts/05_expand_degrees.py --dataset ADE20K_textured_images --dry-run
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, get_output_dir

DEGREE_PATTERN = re.compile(r"(ADE_train_\d+)_textured_degree([\d.]+)")


def main():
    parser = argparse.ArgumentParser(description="Expand with higher-degree variants")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config = load_config(
        str(project_root / "configs" / "base.yaml"),
        str(project_root / "configs" / f"{args.dataset}.yaml"),
    )

    output_dir = get_output_dir(config)
    extract_manifest_path = output_dir / "extract_manifest.json"
    metadata_path = output_dir / config["output"]["metadata_file"]
    images_dir = output_dir / config["output"]["image_dir"]
    textures_mask_dir = output_dir / config["output"]["textures_mask_dir"]

    if not extract_manifest_path.exists():
        print(f"ERROR: {extract_manifest_path} not found. Run extract step first.")
        sys.exit(1)

    with open(extract_manifest_path) as f:
        extracted = json.load(f)

    print(f"Loaded {len(extracted)} extracted images from extract_manifest.json")

    # Convert extracted entries to metadata format (with empty descriptions)
    metadata = []
    for entry in extracted:
        textures = []
        class_names = entry.get("class_names", [])
        mask_paths = entry.get("mask_paths", [])
        for i, mask_path in enumerate(mask_paths):
            name = class_names[i] if i < len(class_names) else f"texture_{i}"
            textures.append({
                "description": "",  # filled by 03_describe.py
                "mask_path": mask_path,
                "class_name": name,
            })
        metadata.append({
            "image_path": entry["image_path"],
            "textures": textures,
            "source_image_id": entry["source_image_id"],
            "id": entry["id"],
            "review_score": entry.get("review_score", 0),
            "overlay_path": entry.get("overlay_path", ""),
            "num_textures": entry.get("num_textures", len(textures)),
        })

    # Load image mapping to find all degree variants
    mapping_path = config["source"].get("image_mapping", "")
    if not mapping_path or not Path(mapping_path).exists():
        print(f"No image_mapping found — skipping degree expansion.")
        # Just save metadata as-is
        if not args.dry_run:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved {len(metadata)} entries to {metadata_path}")
        return

    with open(mapping_path) as f:
        image_mapping = json.load(f)

    # Build base_id → {degree: mapping_entry} index
    base_to_degrees = defaultdict(dict)
    for flat_name, info in image_mapping.items():
        base_id = info["base_ade20k_id"]
        degree = info["degree"]
        base_to_degrees[base_id][degree] = info

    # Find existing entries and their degrees
    existing_ids = set()
    base_passing_degrees = defaultdict(set)

    for entry in metadata:
        img_id = entry.get("source_image_id", entry.get("id", ""))
        existing_ids.add(img_id)
        m = DEGREE_PATTERN.match(img_id)
        if m:
            base_id = m.group(1)
            degree = float(m.group(2))
            base_passing_degrees[base_id].add(degree)

    # For each base image, find min passing degree, add all higher degrees
    new_entries = []
    textured_dir = Path(config["source"]["textured_images_dir"])

    for base_id, passing_degrees in sorted(base_passing_degrees.items()):
        min_degree = min(passing_degrees)
        available = base_to_degrees.get(base_id, {})

        # Find a reference entry for this base to reuse masks
        ref_entry = None
        for entry in metadata:
            eid = entry.get("source_image_id", entry.get("id", ""))
            m2 = DEGREE_PATTERN.match(eid)
            if m2 and m2.group(1) == base_id:
                ref_entry = entry
                break
        if ref_entry is None:
            continue

        for degree, info in sorted(available.items()):
            if degree <= min_degree:
                continue

            new_id = f"{base_id}_textured_degree{degree}"
            if new_id in existing_ids:
                continue

            # Find source image
            flat_image = textured_dir / info["flat_name"]
            if not flat_image.exists():
                src_path = Path(info.get("original_path", ""))
                if not src_path.exists():
                    continue
                flat_image = src_path

            # Copy image to output
            dst_image = images_dir / f"{new_id}.jpg"
            if not args.dry_run and not dst_image.exists():
                shutil.copy2(str(flat_image.resolve()), str(dst_image))

            # Reuse masks from reference entry (same ADE20K annotation)
            new_textures = []
            for tex in ref_entry["textures"]:
                new_textures.append({
                    "description": "",  # filled by 03_describe.py
                    "mask_path": tex["mask_path"],
                    "class_name": tex.get("class_name", ""),
                })

            new_entry = {
                "image_path": str(dst_image),
                "textures": new_textures,
                "source_image_id": new_id,
                "id": new_id,
                "base_ade20k_id": base_id,
                "degree": degree,
                "expanded_from_degree": min_degree,
                "num_textures": len(new_textures),
            }
            new_entries.append(new_entry)
            existing_ids.add(new_id)

    print(f"Extracted images (from pipeline): {len(metadata)}")
    print(f"New higher-degree variants to add: {len(new_entries)}")

    if new_entries:
        degree_dist = Counter(e["degree"] for e in new_entries)
        print(f"\nNew entries by degree:")
        for d in sorted(degree_dist.keys()):
            print(f"  degree {d}: {degree_dist[d]} images")

        base_dist = Counter(e["base_ade20k_id"] for e in new_entries)
        print(f"\nFrom {len(base_dist)} unique base images")

    total = len(metadata) + len(new_entries)
    print(f"\nTotal after expansion: {total} images")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Combine and save as metadata.json
    metadata.extend(new_entries)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved {metadata_path}")

    needs_describe = sum(
        1 for e in metadata
        for t in e.get("textures", [])
        if not t.get("description", "").strip()
    )
    print(f"Entries needing Claude descriptions: {needs_describe}")


if __name__ == "__main__":
    main()
