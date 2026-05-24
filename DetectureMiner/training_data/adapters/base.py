"""Abstract base adapter for dataset-specific logic."""

from abc import ABC, abstractmethod
from pathlib import Path


class DatasetAdapter(ABC):
    """
    Each dataset source (ADE20K, COCO, etc.) implements this interface
    to handle its specific metadata format, path conventions, and scoring.
    """

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def get_score_command(self) -> list[str]:
        """Return the CLI command to run Detecture scoring."""
        ...

    @abstractmethod
    def load_extraction_metadata(self) -> list[dict]:
        """Load the existing Detecture extraction metadata."""
        ...

    @abstractmethod
    def get_image_path(self, entry: dict) -> str:
        """Get the absolute image path for a transition entry."""
        ...

    @abstractmethod
    def get_mask_paths(self, entry: dict) -> list[str]:
        """Get absolute mask paths (one per texture region)."""
        ...

    @abstractmethod
    def get_existing_descriptions(self, entry: dict) -> list[str]:
        """Get existing VLM-generated descriptions as hints."""
        ...

    @abstractmethod
    def get_overlay_path(self, entry: dict) -> str | None:
        """Get overlay visualization path, if available."""
        ...

    @abstractmethod
    def get_source_image_id(self, entry: dict) -> str:
        """Get the source image ID (for grouping transitions from same image)."""
        ...

    @abstractmethod
    def get_entry_id(self, entry: dict) -> str:
        """Get a unique ID for this transition entry."""
        ...
