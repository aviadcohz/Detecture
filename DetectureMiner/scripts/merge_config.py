#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency PyYAML. Activate the project environment and run: pip install -r requirements.txt"
    ) from exc


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            if k in out:
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    if isinstance(base, list) and isinstance(override, list):
        return list(override)
    return override


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge base config with profile config")
    parser.add_argument("--base", required=True, help="Base YAML config path")
    parser.add_argument("--profile", required=True, help="Profile YAML config path")
    parser.add_argument("--out", required=True, help="Output merged YAML path")
    args = parser.parse_args()

    base_path = Path(args.base)
    profile_path = Path(args.profile)
    out_path = Path(args.out)

    base_cfg = load_yaml(base_path)
    profile_cfg = load_yaml(profile_path)

    merged = deep_merge(base_cfg, profile_cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=False), encoding="utf-8")

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
