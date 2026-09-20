"""Run bookkeeping: output directory naming, per-split counts, and the evaluated-subset header."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence


def run_dir(out_dir: str, task: str, model: str, n_val: int, n_test: int, suffix: str = "") -> Path:
    return Path(out_dir) / task / f"{model.replace('/', '__')}{suffix}_val{n_val}_test{n_test}"


def split_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    correct = sum(r["correct"] for r in records)
    return {
        "n": n,
        "correct": correct,
        "incorrect": n - correct,
        "no_answer": sum(r["prediction"] is None for r in records),
        "truncated": sum(r["finish_reason"] == "length" for r in records),
        "implicit_answers": sum(r["prediction"] is not None and not r["explicit_answer"] for r in records),
        "accuracy": correct / n,
    }


def header_lines(summaries: Dict[str, Dict[str, Any]], records: Dict[str, Sequence[Dict[str, Any]]],
                 answered_only: bool) -> List[str]:
    lines = [f"{split}: " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items())
             for split, s in summaries.items()]
    test = records["test"]
    evaluated = [r for r in test if r["prediction"] is not None] if answered_only else list(test)
    correct = sum(r["correct"] for r in evaluated)
    lines.append(f"\nconfidence evaluation: {'answered_only' if answered_only else 'all_examples'}")
    lines.append(f"evaluated={len(evaluated)} correct={correct} incorrect={len(evaluated) - correct} "
                 f"excluded={len(test) - len(evaluated)}")
    validation = [r for r in records.get("validation", []) if r["prediction"] is not None or not answered_only]
    labels = {r["correct"] for r in validation}
    if not validation:
        lines.append("calibration: none (no validation split)")
    elif len(labels) < 2:
        missing = "incorrect" if True in labels else "correct"
        lines.append(f"WARNING: calibration unavailable - validation has no {missing} examples.")
    return lines


def timing_line(timings: Dict[str, Dict[str, float]]) -> str:
    parts = [f"{stage}={t['seconds']:.0f}s (cache {t['cache_hits']}/{t['cache_hits'] + t['cache_misses']})"
             for stage, t in timings.items()]
    return "wall time incl. cache reuse: " + " ".join(parts)
