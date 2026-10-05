"""Small .env loader for quick-start scripts.

This avoids requiring python-dotenv while keeping credentials and local paths
out of commands and source files.
"""

from __future__ import annotations

import os
from pathlib import Path


QUICK_START = Path(__file__).resolve().parents[1]


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_quickstart_env(*, override: bool = False) -> Path:
    env_path = QUICK_START / ".env"
    if not env_path.exists():
        return env_path

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value)
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path

