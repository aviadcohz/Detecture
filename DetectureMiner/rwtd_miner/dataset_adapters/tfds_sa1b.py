from __future__ import annotations

from pathlib import Path
from typing import Iterable

from rwtd_miner.dataset_adapters.base import BaseDatasetAdapter, ImageRecord
from rwtd_miner.utils.logging import get_logger


class TFDSSA1BAdapter(BaseDatasetAdapter):
    def __init__(self, input_root: Path, config: dict, dataset_name: str = "segment_anything"):
        super().__init__(input_root, config)
        self.log = get_logger("TFDSSA1BAdapter")
        self.tfds_name = dataset_name

    def dataset_name(self) -> str:
        return f"tfds:{self.tfds_name}"

    def discover_images(self) -> Iterable[ImageRecord]:
        try:
            import tensorflow_datasets as tfds
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("TFDS adapter requested but tensorflow_datasets is unavailable") from exc

        ds = tfds.load(self.tfds_name, split="train", shuffle_files=False)
        for idx, ex in enumerate(tfds.as_numpy(ds)):
            image_id = str(ex.get("image_id", idx))
            # TFDS images are not file-backed here; keep pseudo-path for manifest compatibility.
            pseudo = Path(f"tfds://{self.tfds_name}/{idx}")
            h, w = ex["image"].shape[:2]
            yield ImageRecord(
                image_id=image_id,
                image_path=pseudo,
                annotation_ref=None,
                file_size_bytes=int(ex["image"].nbytes),
                width=int(w),
                height=int(h),
            )
