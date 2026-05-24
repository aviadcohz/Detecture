from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ImageRecord:
    image_id: str
    image_path: Path
    annotation_ref: str | None
    file_size_bytes: int
    width: int | None
    height: int | None


class BaseDatasetAdapter:
    def __init__(self, input_root: Path, config: dict):
        self.input_root = input_root
        self.config = config

    def discover_images(self) -> Iterable[ImageRecord]:
        raise NotImplementedError

    def dataset_name(self) -> str:
        return "base"
