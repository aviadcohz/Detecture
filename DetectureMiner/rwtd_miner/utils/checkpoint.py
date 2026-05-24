from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rwtd_miner.utils.io import read_json, write_json


@dataclass
class Checkpoint:
    path: Path
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        if path.exists():
            try:
                data = read_json(path)
            except Exception:
                data = {}
        else:
            data = {}
        return cls(path=path, data=data)

    def mark_stage_done(self, stage_name: str, extra: dict[str, Any] | None = None) -> None:
        stages = self.data.setdefault("stages", {})
        stages[stage_name] = {"done": True, **(extra or {})}
        self.save()

    def is_stage_done(self, stage_name: str) -> bool:
        stages = self.data.get("stages", {})
        return bool(stages.get(stage_name, {}).get("done", False))

    def save(self) -> None:
        write_json(self.path, self.data)
