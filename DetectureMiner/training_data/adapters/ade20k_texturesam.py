"""ADE20K TextureSAM adapter for texture-augmented ADE20K images.

Works with images from TextureSAM's ADE20K_0.3 dataset (texture-augmented
ADE20K images at eta=0.3) mapped back to original ADE20K semantic annotations.

Fully independent — scores its own images without depending on ade20k run.
"""

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

_TEXTURED_PATTERN = re.compile(r"(ADE_train_\d+)_textured_degree(\d+)_(\d+)")


class ADE20KTextureSAMAdapter(DatasetAdapter):
    """Adapter for TextureSAM texture-augmented ADE20K images."""

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

    def get_score_command(self) -> list[str]:
        """Score the original ADE20K images, output to OWN scoring dir."""
        cfg = self.config
        detecture_root = cfg["score"]["detecture_root"]

        cmd = [
            "python", "-m", "rwtd_miner.cli", "ade20k_full",
            "--config", f"{detecture_root}/config.yaml",
            "--out", str(Path(cfg["output"]["base_dir"]) / cfg["dataset"] / "scoring"),
            "--ade_root", str(self._ade_root),
            "--selected_min", str(cfg["score"]["selected_min"]),
            "--borderline_min", str(cfg["score"]["borderline_min"]),
            "--review_limit", "0",  # 0 = no limit, score ALL images
        ]
        if cfg["score"].get("enable_clip"):
            cmd.append("--enable_clip")
        if cfg["score"].get("enable_vlm"):
            cmd.extend(["--enable_vlm", "--vlm_backend", "hf_blip_vqa", "--vlm_device", "auto"])
        return cmd

    def _load_scores(self, scoring_dir: Path) -> dict[str, float]:
        """Load scores from OWN scoring dir → base_id → score map."""
        import pandas as pd

        parquet_files = list(scoring_dir.glob("**/*.parquet"))
        if parquet_files:
            df = pd.read_parquet(parquet_files[0])
        else:
            csv_files = list(scoring_dir.glob("**/*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
            else:
                return {}

        scores = {}
        for _, row in df.iterrows():
            image_id = str(row.get("image_id", ""))
            # Convert "training_ADE_train_XXXXX" → "ADE_train_XXXXX"
            base_id = image_id.replace("training_", "") if image_id.startswith("training_") else image_id
            scores[base_id] = float(row.get("review_score", 0))
        return scores

    def load_scored_images(self, scoring_dir: Path) -> list[dict]:
        """Load scores from OWN scoring dir and map to textured variants."""
        scores = self._load_scores(scoring_dir)

        if not scores:
            print(f"WARNING: No scores found in {scoring_dir}")
            print("Run scoring first: python scripts/01_score.py --dataset ade20k_texturesam")
            return []

        print(f"Loaded {len(scores)} base image scores")

        entries = []
        matched = 0
        unmatched = 0
        for img_path in sorted(self._textured_dir.glob("*.jpg")):
            m = _TEXTURED_PATTERN.match(img_path.stem)
            if not m:
                continue
            base_id = m.group(1)
            ann_path = self._annotations_dir / f"{base_id}.png"
            if not ann_path.exists():
                continue

            score = scores.get(base_id, -1.0)
            if score < 0:
                unmatched += 1
                continue

            matched += 1
            entries.append({
                "image_id": img_path.stem,
                "review_score": score,
                "image_path": str(img_path),
                "annotation_ref": str(ann_path),
                "base_ade20k_id": base_id,
            })

        print(f"Matched to scores: {matched}, no score available: {unmatched}")
        return entries

    def get_image_path(self, entry: dict) -> str:
        return entry["image_path"]

    def get_annotation_path(self, entry: dict) -> str:
        p = entry.get("annotation_ref", "")
        if p and Path(p).exists():
            return p
        base_id = entry.get("base_ade20k_id", "")
        if not base_id:
            m = _TEXTURED_PATTERN.match(entry["image_id"])
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
