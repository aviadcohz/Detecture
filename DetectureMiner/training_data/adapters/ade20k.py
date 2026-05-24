"""ADE20K adapter for raw ADE20K dataset (ADEChallengeData2016).

Works directly on raw images + semantic annotations.
Creates one mask per ADE20K class (merging all spatial instances),
filtered by minimum area. This matches the "Annotation mask visualization"
from the Detecture gallery.
"""

import re
import numpy as np
import cv2
from pathlib import Path

from .base import DatasetAdapter


# --- ADE20K class name parsing ---

def _parse_object_info(path: Path) -> dict[int, str]:
    """Parse ADE20K objectInfo150.txt → {class_id: class_name}."""
    out = {}
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("Idx"):
            continue
        parts = s.split("\t")
        if parts and parts[0].strip().isdigit():
            idx = int(parts[0].strip())
            name = parts[4].strip() if len(parts) >= 5 else parts[1].strip()
            out[idx] = name.lower()
    return out


# Fixed color palette for up to 20 texture regions (BGR format for OpenCV)
REGION_COLORS = [
    (255, 0, 0),    # blue
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (128, 0, 0),    # dark blue
    (0, 128, 0),    # dark green
    (0, 0, 128),    # dark red
    (128, 128, 0),  # teal
    (0, 128, 128),  # olive
    (128, 0, 128),  # purple
    (255, 128, 0),  # light blue
    (128, 255, 0),  # light green
    (0, 128, 255),  # orange
    (255, 0, 128),  # pink-blue
    (128, 255, 128),# light mint
    (255, 128, 128),# light cyan
    (128, 128, 255),# light salmon
    (200, 200, 0),  # dark cyan
]


def extract_class_masks(
    annotation_path: str,
    class_names: dict[int, str],
    min_region_frac: float = 0.01,
    max_regions: int = 10,
) -> list[tuple[np.ndarray, int, str]]:
    """
    Extract masks from ADE20K annotation — one mask per CLASS (all instances merged).

    Returns list of (binary_mask_uint8_0_255, class_id, class_name),
    sorted by area descending, up to max_regions.
    Ignores class 0 (background/unlabeled).
    """
    ann = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
    if ann is None:
        return []

    h, w = ann.shape
    total = h * w

    # Get all class IDs present in this annotation
    unique_ids = np.unique(ann)

    regions = []  # (area, mask, class_id, class_name)
    for cid in unique_ids:
        cid = int(cid)
        if cid == 0:
            continue  # skip background

        # Merge ALL pixels of this class into one mask
        mask = (ann == cid).astype(np.uint8) * 255
        area = int((mask > 0).sum())
        frac = area / total

        if frac < min_region_frac:
            continue

        name = class_names.get(cid, f"class_{cid}")
        regions.append((area, mask, cid, name))

    # Sort by area descending
    regions.sort(key=lambda x: x[0], reverse=True)
    return [(mask, cid, name) for _, mask, cid, name in regions[:max_regions]]


def create_index_mask(masks: list[np.ndarray], h: int, w: int) -> np.ndarray:
    """Create a single index mask from list of binary masks.
    Pixel value = region index (1-based), 0 = uncovered/dustbin."""
    index_mask = np.zeros((h, w), dtype=np.uint8)
    for i, mask in enumerate(masks):
        index_mask[mask > 127] = i + 1
    return index_mask


def create_colored_visualization(index_mask: np.ndarray) -> np.ndarray:
    """Create a colored annotation mask visualization (like the gallery)."""
    h, w = index_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for region_id in range(1, int(index_mask.max()) + 1):
        color = REGION_COLORS[(region_id - 1) % len(REGION_COLORS)]
        vis[index_mask == region_id] = color
    return vis


def create_overlay(image: np.ndarray, index_mask: np.ndarray) -> np.ndarray:
    """Create overlay of colored masks on original image."""
    vis = create_colored_visualization(index_mask)
    overlay = image.copy()
    covered = index_mask > 0
    overlay[covered] = cv2.addWeighted(image, 0.4, vis, 0.6, 0)[covered]

    # Draw contours between regions
    for region_id in range(1, int(index_mask.max()) + 1):
        region_mask = (index_mask == region_id).astype(np.uint8)
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = REGION_COLORS[(region_id - 1) % len(REGION_COLORS)]
        cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


class ADE20KAdapter(DatasetAdapter):
    """Adapter for raw ADE20K dataset."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._ade_root = Path(config["source"]["ade_root"])
        self._class_names = _parse_object_info(self._ade_root / "objectInfo150.txt")
        self._images_dir = self._ade_root / "images" / "training"
        self._annotations_dir = self._ade_root / "annotations" / "training"

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    def get_score_command(self) -> list[str]:
        cfg = self.config
        detecture_root = cfg["score"]["detecture_root"]
        cli_cmd = cfg["score"]["cli_command"]

        cmd = [
            "python", "-m", "rwtd_miner.cli", cli_cmd,
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

    def load_scored_images(self, scoring_dir: Path) -> list[dict]:
        """Load scored images from Detecture scoring output."""
        import pandas as pd

        parquet_files = list(scoring_dir.glob("**/*.parquet"))
        if parquet_files:
            df = pd.read_parquet(parquet_files[0])
        else:
            csv_files = list(scoring_dir.glob("**/*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
            else:
                return []

        entries = []
        for _, row in df.iterrows():
            image_id = str(row.get("image_id", ""))
            entries.append({
                "image_id": image_id,
                "review_score": float(row.get("review_score", 0)),
                "image_path": str(row.get("image_path", "")),
                "annotation_ref": str(row.get("annotation_ref", "")),
            })
        return entries

    def load_extraction_metadata(self) -> list[dict]:
        raise NotImplementedError("Use load_scored_images() instead")

    def _strip_prefix(self, image_id: str) -> str:
        """Convert 'training_ADE_train_XXXXX' → 'ADE_train_XXXXX'."""
        if image_id.startswith("training_"):
            return image_id[len("training_"):]
        return image_id

    def get_image_path(self, entry: dict) -> str:
        p = entry.get("image_path", "")
        if p and Path(p).exists():
            return p
        base = self._strip_prefix(entry["image_id"])
        return str(self._images_dir / f"{base}.jpg")

    def get_annotation_path(self, entry: dict) -> str:
        p = entry.get("annotation_ref", "")
        if p and Path(p).exists():
            return p
        base = self._strip_prefix(entry["image_id"])
        return str(self._annotations_dir / f"{base}.png")

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
