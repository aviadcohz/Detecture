"""ADE20K TextureSAM Full adapter — all 222K images (all degrees 0.0-1.0).

Feeds ALL textured variants as independent images into the pipeline.
Each variant gets its own CLIP score and VLM score.

Stage A (geometry) runs once per unique annotation (20K), then scores
are propagated to all 11 degree variants. Stages B (CLIP) and D (VLM)
run on every individual textured image.
"""

import csv
import json
import re
from pathlib import Path

from .base import DatasetAdapter
from .ade20k import (
    _parse_object_info,
    extract_class_masks,
    create_index_mask,
    create_colored_visualization,
    create_overlay,
)

DEGREE_PATTERN = re.compile(r"(ADE_train_\d+)_textured_degree([\d.]+)")


class ADE20KTextureSAMFullAdapter(DatasetAdapter):
    """Adapter for full TextureSAM dataset — all degrees as independent images."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._ade_root = Path(config["source"]["ade_root"])
        self._class_names = _parse_object_info(self._ade_root / "objectInfo150.txt")
        self._annotations_dir = self._ade_root / "annotations" / "training"
        self._textured_dir = Path(config["source"]["textured_images_dir"])
        self._mapping = self._load_mapping(config["source"].get("image_mapping"))

    def _load_mapping(self, mapping_path: str | None) -> dict:
        if mapping_path and Path(mapping_path).exists():
            with open(mapping_path) as f:
                return json.load(f)
        return {}

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    def _generate_manifest_csv(self, output_dir: Path) -> str:
        """
        Generate a CSV manifest with ALL textured variants and their annotations.
        This is passed to the Detecture CLI via --custom_images_csv.

        Returns path to the generated CSV.
        """
        manifest_path = output_dir / "scoring" / "custom_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        n_written = 0
        n_skipped = 0
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "image_path", "annotation_ref"])
            writer.writeheader()

            for flat_name, info in sorted(self._mapping.items()):
                base_id = info["base_ade20k_id"]
                ann_path = self._annotations_dir / f"{base_id}.png"
                if not ann_path.exists():
                    n_skipped += 1
                    continue

                img_path = self._textured_dir / flat_name
                if not img_path.exists():
                    n_skipped += 1
                    continue

                writer.writerow({
                    "image_id": Path(flat_name).stem,
                    "image_path": str(img_path.resolve()),
                    "annotation_ref": str(ann_path),
                })
                n_written += 1

        print(f"Generated manifest: {n_written} images, {n_skipped} skipped → {manifest_path}")
        return str(manifest_path)

    def get_score_command(self) -> list[str]:
        """
        Score ALL textured variants via custom manifest.

        Stage A (geometry) runs on unique annotations only.
        Stages B (CLIP) and D (VLM) run on every image.
        """
        cfg = self.config
        detecture_root = cfg["score"]["detecture_root"]
        # Must be absolute — CLI runs with cwd=detecture_root
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / cfg["output"]["base_dir"] / cfg["dataset"]

        # Generate the manifest CSV with all variants
        manifest_csv = self._generate_manifest_csv(output_dir)

        cmd = [
            "python", "-m", "rwtd_miner.cli", "ade20k_full",
            "--config", f"{detecture_root}/config.yaml",
            "--out", str(output_dir / "scoring"),
            "--ade_root", str(self._ade_root),
            "--selected_min", str(cfg["score"]["selected_min"]),
            "--borderline_min", str(cfg["score"]["borderline_min"]),
            "--review_limit", "0",
            "--custom_images_csv", manifest_csv,
            "--enable_clip",
            "--enable_vlm",
            "--vlm_top_n", "222310",  # score all images (no limit)
            "--skip_download",
        ]
        return cmd

    def _load_scores(self, scoring_dir: Path) -> dict[str, float]:
        """Load per-image scores from scoring output."""
        import pandas as pd

        # Try parquet first, then CSV
        for pattern in ["manifests/*.parquet", "**/*.parquet", "manifests/*.csv", "**/*.csv"]:
            files = sorted(scoring_dir.glob(pattern))
            if files:
                if files[0].suffix == ".parquet":
                    df = pd.read_parquet(files[0])
                else:
                    df = pd.read_csv(files[0])
                break
        else:
            return {}

        scores = {}
        for _, row in df.iterrows():
            image_id = str(row.get("image_id", ""))
            scores[image_id] = float(row.get("review_score", 0))
        return scores

    def load_scored_images(self, scoring_dir: Path) -> list[dict]:
        """
        Load per-image scores (each variant scored independently).
        """
        scores = self._load_scores(scoring_dir)

        if not scores:
            print(f"WARNING: No scores found in {scoring_dir}")
            print(f"Run scoring first: python scripts/01_score.py --dataset {self.config['dataset']}")
            return []

        print(f"Loaded {len(scores)} individually scored images")

        entries = []
        for flat_name, info in sorted(self._mapping.items()):
            image_id = Path(flat_name).stem
            if image_id not in scores:
                continue

            base_id = info["base_ade20k_id"]
            ann_path = self._annotations_dir / f"{base_id}.png"
            if not ann_path.exists():
                continue

            entries.append({
                "image_id": image_id,
                "review_score": scores[image_id],
                "image_path": str(self._textured_dir / flat_name),
                "annotation_ref": str(ann_path),
                "base_ade20k_id": base_id,
                "degree": info["degree"],
            })

        print(f"Total entries with scores: {len(entries)}")
        return entries

    def get_image_path(self, entry: dict) -> str:
        return entry["image_path"]

    def get_annotation_path(self, entry: dict) -> str:
        p = entry.get("annotation_ref", "")
        if p and Path(p).exists():
            return p
        base_id = entry.get("base_ade20k_id", "")
        if not base_id:
            m = DEGREE_PATTERN.match(entry["image_id"])
            if m:
                base_id = m.group(1)
        return str(self._annotations_dir / f"{base_id}.png")

    def load_extraction_metadata(self) -> list[dict]:
        raise NotImplementedError("Use load_scored_images() instead")

    def get_mask_paths(self, entry: dict) -> list[str]:
        raise NotImplementedError("Masks are generated during extraction")

    def get_existing_descriptions(self, entry: dict) -> list[str]:
        return []

    def get_overlay_path(self, entry: dict) -> str | None:
        return None

    def get_source_image_id(self, entry: dict) -> str:
        return entry["image_id"]

    def get_entry_id(self, entry: dict) -> str:
        return entry["image_id"]
