"""Shared utilities for the data preparation pipeline."""

import yaml
import cv2
import numpy as np
from pathlib import Path


def load_config(base_path: str, dataset_path: str) -> dict:
    """Load and merge base config with dataset-specific config."""
    with open(base_path) as f:
        base = yaml.safe_load(f)
    with open(dataset_path) as f:
        dataset = yaml.safe_load(f)

    # Deep merge: dataset overrides base
    merged = _deep_merge(base, dataset)
    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def fix_path(path: str, prefix_from: str, prefix_to: str) -> str:
    """Fix path prefix (e.g., /datasets/ -> ~/datasets/)."""
    if path.startswith(prefix_from):
        return prefix_to + path[len(prefix_from):]
    return path


def validate_mask(mask_path: str, threshold: int = 127) -> bool:
    """Check that a mask file is a valid binary mask (only 0 and 255 values)."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    unique_vals = set(np.unique(mask))
    return unique_vals.issubset({0, 255})


def load_mask(mask_path: str) -> np.ndarray | None:
    """Load a grayscale mask, return None if file doesn't exist."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask


def get_output_dir(config: dict) -> Path:
    """Get the dataset output directory from config."""
    project_root = Path(__file__).parent
    base_dir = config["output"]["base_dir"]
    dataset = config["dataset"]
    return project_root / base_dir / dataset


def ensure_dirs(output_dir: Path, config: dict) -> dict[str, Path]:
    """Create output subdirectories and return their paths."""
    dirs = {
        "images": output_dir / config["output"]["image_dir"],
        "masks": output_dir / config["output"]["mask_dir"],
        "textures_mask": output_dir / config["output"]["textures_mask_dir"],
        "overlays": output_dir / config["output"]["overlay_dir"],
        "scoring": output_dir / "scoring",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
