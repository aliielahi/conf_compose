"""Configuration helpers for MATU command-line workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML configs require PyYAML. Install with `pip install PyYAML`.") from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a mapping: {path}")
    return data


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML or JSON config file and expand environment variables."""
    if path is None:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = _load_yaml(config_path)
    elif suffix == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Config file must contain a mapping: {config_path}")
    else:
        raise ValueError(f"Unsupported config extension {suffix!r}; use .yaml, .yml, or .json")

    return _expand_env(data)


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section {name!r} must be a mapping")
    return value


def choose(cli_value: Any, config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Use a CLI value when provided, otherwise fall back to config/default."""
    if cli_value is not None:
        return cli_value
    return config.get(key, default)
