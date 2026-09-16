"""Run bookkeeping: output directory naming and per-split accuracy/truncation summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence


def run_dir(out_dir: str, task: str, model: str, n_val: int, n_test: int) -> Path:
    return Path(out_dir) / task / f"{model.replace('/', '__')}_val{n_val}_test{n_test}"


def split_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in records) / n,
        "truncated": sum(r["finish_reason"] == "length" for r in records) / n,
        "implicit_answers": sum(r["prediction"] is not None and not r["explicit_answer"] for r in records) / n,
        "no_answer": sum(r["prediction"] is None for r in records) / n,
    }
