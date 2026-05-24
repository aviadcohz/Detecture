from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from rwtd_miner.dataset_adapters.base import BaseDatasetAdapter, ImageRecord
from rwtd_miner.utils.image_utils import safe_image_size
from rwtd_miner.utils.logging import get_logger


class LocalSA1BAdapter(BaseDatasetAdapter):
    def __init__(self, input_root: Path, config: dict):
        super().__init__(input_root, config)
        self.log = get_logger("LocalSA1BAdapter")
        self.images_dir = self.input_root / "images"
        self.annotations_dir = self.input_root / "annotations"
        idx_cfg = self.config.get("index", {})
        self.image_exts = {x.lower() for x in idx_cfg.get("image_extensions", [".jpg", ".jpeg", ".png", ".webp"])}
        self.annotation_exts = {
            x.lower() for x in idx_cfg.get("annotation_extensions", [".json", ".mat", ".png"])
        }
        self.read_dims = bool(idx_cfg.get("read_image_dimensions", True))
        self.scan_annotation_shards = bool(idx_cfg.get("annotation_scan_shards", True))
        self.shard_scan_max_files = int(idx_cfg.get("shard_scan_max_files", 500))

    def dataset_name(self) -> str:
        return "sa1b_local"

    def _build_direct_annotation_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if not self.annotations_dir.exists():
            return out
        for p in self.annotations_dir.rglob("*"):
            if (not p.is_file()) or (p.suffix.lower() not in self.annotation_exts):
                continue
            stem = p.stem
            out.setdefault(stem, str(p))
        return out

    @staticmethod
    def _extract_image_key_from_entry(entry: dict) -> str | None:
        # Prefer filename-derived keys to preserve zero padding (e.g., COCO ids).
        for key in ("file_name", "image_path", "image"):
            value = entry.get(key)
            if not value:
                continue
            return Path(str(value)).stem
        for key in ("image_id", "id", "sa_id"):
            value = entry.get(key)
            if value is not None:
                return str(value)
        image = entry.get("image")
        if isinstance(image, dict):
            fname = image.get("file_name")
            if fname:
                return Path(str(fname)).stem
            iid = image.get("image_id") or image.get("id")
            if iid is not None:
                return str(iid)
        return None

    def _build_shard_annotation_map(self, direct_map: dict[str, str]) -> dict[str, str]:
        shard_map: dict[str, str] = {}
        if not self.annotations_dir.exists() or not self.scan_annotation_shards:
            return shard_map

        # Scan shard-like JSON files first (typically larger files with many annotations).
        # The previous stem-based filter could skip every JSON file, leaving shard_map empty.
        shard_files = list(self.annotations_dir.rglob("*.json"))
        shard_files.sort(key=lambda p: (-p.stat().st_size, str(p)))
        if self.shard_scan_max_files > 0:
            shard_files = shard_files[: self.shard_scan_max_files]

        for path in tqdm(shard_files, desc="scan_annotation_shards", unit="file"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            if isinstance(payload, list):
                for idx, entry in enumerate(payload):
                    if not isinstance(entry, dict):
                        continue
                    image_key = self._extract_image_key_from_entry(entry)
                    if image_key and image_key not in direct_map:
                        shard_map.setdefault(image_key, f"{path}::{idx}")
                continue

            if isinstance(payload, dict):
                if isinstance(payload.get("images"), list) and isinstance(payload.get("annotations"), list):
                    images = payload["images"]
                    # COCO-style mapping.
                    for entry in images:
                        if not isinstance(entry, dict):
                            continue
                        image_key = self._extract_image_key_from_entry(entry)
                        if image_key and image_key not in direct_map:
                            shard_map.setdefault(image_key, str(path))
                elif isinstance(payload.get("data"), list):
                    for idx, entry in enumerate(payload["data"]):
                        if not isinstance(entry, dict):
                            continue
                        image_key = self._extract_image_key_from_entry(entry)
                        if image_key and image_key not in direct_map:
                            shard_map.setdefault(image_key, f"{path}::{idx}")
        self.log.info("Shard map indexed for %s images", len(shard_map))
        return shard_map

    def discover_images(self) -> Iterable[ImageRecord]:
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing images directory: {self.images_dir}")

        direct_map = self._build_direct_annotation_map()
        shard_map = self._build_shard_annotation_map(direct_map)

        image_paths = [p for p in self.images_dir.rglob("*") if p.is_file() and p.suffix.lower() in self.image_exts]
        image_paths.sort()

        for p in tqdm(image_paths, desc="index_images", unit="img"):
            image_id = p.stem
            ann_ref = direct_map.get(image_id) or shard_map.get(image_id)
            file_size_bytes = int(p.stat().st_size)
            w, h = (None, None)
            if self.read_dims:
                w, h = safe_image_size(p)
            yield ImageRecord(
                image_id=image_id,
                image_path=p,
                annotation_ref=ann_ref,
                file_size_bytes=file_size_bytes,
                width=w,
                height=h,
            )
