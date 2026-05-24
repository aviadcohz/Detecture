from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_table(df: pd.DataFrame, path_base: Path, write_csv: bool = True) -> None:
    ensure_dir(path_base.parent)
    parquet_path = path_base.with_suffix(".parquet")
    csv_path = path_base.with_suffix(".csv")
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        # Parquet might fail if pyarrow is missing; continue with CSV.
        pass
    if write_csv:
        df.to_csv(csv_path, index=False)


def read_table(path_base: Path) -> pd.DataFrame:
    parquet_path = path_base.with_suffix(".parquet")
    csv_path = path_base.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()
