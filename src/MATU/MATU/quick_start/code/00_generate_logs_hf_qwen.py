"""Step 0B: generate example HF/Qwen conversation logs.

This is an optional data-collection step. The quick-start folder already
contains reference logs under ``quick_start/data``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from env_utils import load_quickstart_env


QUICK_START = Path(__file__).resolve().parents[1]
REPO_ROOT = QUICK_START.parent
OUT_DIR = QUICK_START / "generated"


def main() -> None:
    load_quickstart_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "examples" / "generate_logs_hf_qwen.py"),
        "--data_root",
        str(QUICK_START / "data" / "MATH" / "test"),
        "--category",
        "algebra",
        "--max_items",
        "5",
        "--runs",
        "10",
        "--model",
        os.environ.get("MATU_QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "--out",
        str(OUT_DIR / "conversation_logs_hf_qwen.json"),
        "--embedding_out_dir",
        str(OUT_DIR / "embeddings_from_hf_qwen_generation"),
        "--embedding_model",
        os.environ.get("MATU_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
    ]
    cache_dir = os.environ.get("MATU_MODEL_CACHE")
    if cache_dir:
        cmd.extend(["--cache_dir", cache_dir])
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


if __name__ == "__main__":
    main()
