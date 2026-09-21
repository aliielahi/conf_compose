"""Evaluate composition methods: ranking and reliability on the scored subset, paired against a reference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from conf_compose.utils.metrics import auarc, auroc, brier, ece, nll

SEED = 20260920


@dataclass
class Row:
    """One method's predictions over the examples where it produced a score."""
    method: str
    example_ids: List[str]
    scores: List[float]
    correct: List[float]
    is_probability: bool
    answers: List[Optional[str]]
    total: int
    logits: Optional[List[Optional[float]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def evaluate(rows: Sequence[Row], reference: str, n_boot: int = 2000,
             intercepts: Optional[Dict[str, float]] = None,
             extra_references: Sequence[str] = ()) -> Dict[str, Dict[str, Any]]:
    report: Dict[str, Dict[str, Any]] = {}
    by_method = {row.method: row for row in rows}
    for row in rows:
        scores, correct = np.array(row.scores, dtype=float), np.array(row.correct, dtype=float)
        entry: Dict[str, Any] = {"n": len(scores), "coverage": len(scores) / row.total if row.total else 0.0,
                                 "errors": int((1 - correct).sum()),
                                 "accuracy": float(correct.mean()) if len(correct) else float("nan"), **row.meta}
        if len(scores) >= 2 and 0 < correct.sum() < len(correct):
            entry.update({"auroc": auroc(scores, correct), "auarc": auarc(scores, correct)})
            if row.is_probability:
                entry.update({"brier": brier(scores, correct), "ece": ece(scores, correct),
                              "nll": nll(scores, correct)})
            if intercepts and row.method in intercepts:
                calibrated = _apply_intercept(_logits_of(row), intercepts[row.method])
                entry.update({"cal_brier": brier(calibrated, correct), "cal_ece": ece(calibrated, correct),
                              "cal_nll": nll(calibrated, correct)})
        for name in [reference, *extra_references]:
            if row.method != name and name in by_method:
                paired = _paired(by_method[name], row, n_boot)
                suffix = "" if name == reference else f"_vs_{name}"
                entry.update({f"{key}{suffix}": value for key, value in paired.items()})
        report[row.method] = entry
    return report


def fit_intercepts(rows: Sequence[Row]) -> Dict[str, float]:
    """One calibration intercept per method, slope fixed at one, fitted by bisection on validation."""
    intercepts: Dict[str, float] = {}
    for row in rows:
        correct = np.array(row.correct, dtype=float)
        if len(row.scores) < 2 or not 0 < correct.sum() < len(correct):
            continue
        z = _logits_of(row)
        low, high = -60.0, 60.0
        for _ in range(200):
            middle = (low + high) / 2
            if float(np.sum(_apply_intercept(z, middle) - correct)) > 0:
                high = middle
            else:
                low = middle
            if high - low < 1e-8:
                break
        intercepts[row.method] = (low + high) / 2
    return intercepts


def best_by_validation(rows: Sequence[Row], report: Dict[str, Dict[str, Any]], prefix: str) -> str:
    """The strongest source by validation AUROC, selected without ever touching test."""
    options = [(report.get(row.method, {}).get("auroc") or 0.0, row.method)
               for row in rows if row.method.startswith(prefix)]
    return max(options)[1] if options else ""


def _paired(reference: Row, row: Row, n_boot: int) -> Dict[str, Any]:
    """Bootstrap the AUROC difference over shared question ids, each method against its own labels."""
    shared = sorted(set(reference.example_ids) & set(row.example_ids))
    if len(shared) < 10:
        return {}
    ref_index = {example: i for i, example in enumerate(reference.example_ids)}
    own_index = {example: i for i, example in enumerate(row.example_ids)}
    ref_scores = np.array([reference.scores[ref_index[e]] for e in shared])
    ref_correct = np.array([reference.correct[ref_index[e]] for e in shared], dtype=float)
    own_scores = np.array([row.scores[own_index[e]] for e in shared])
    own_correct = np.array([row.correct[own_index[e]] for e in shared], dtype=float)
    if not (0 < own_correct.sum() < len(own_correct) and 0 < ref_correct.sum() < len(ref_correct)):
        return {}
    deltas, undefined = [], 0
    generator = np.random.default_rng(SEED)
    for sample in generator.integers(0, len(shared), (n_boot, len(shared))):
        own_labels, ref_labels = own_correct[sample], ref_correct[sample]
        if 0 < own_labels.sum() < len(own_labels) and 0 < ref_labels.sum() < len(ref_labels):
            deltas.append(auroc(own_scores[sample], own_labels) - auroc(ref_scores[sample], ref_labels))
        else:
            undefined += 1
    interval = np.quantile(deltas, [0.025, 0.975]) if deltas else (float("nan"), float("nan"))
    return {"delta_auroc": auroc(own_scores, own_correct) - auroc(ref_scores, ref_correct),
            "delta_ci": [float(interval[0]), float(interval[1])], "paired_n": len(shared),
            "undefined_boot": undefined}


def _logits_of(row: Row) -> np.ndarray:
    """Stored pre-sigmoid scores when a method kept them, otherwise the clipped logit of its probability."""
    if row.logits is not None and all(value is not None for value in row.logits):
        return np.array(row.logits, dtype=float)
    scores = np.array(row.scores, dtype=float)
    if not row.is_probability:
        return scores
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def _apply_intercept(logits: np.ndarray, intercept: float) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(logits + intercept, -60, 60)))


def format_table(report: Dict[str, Dict[str, Any]], reference: str, accuracy: bool = False,
                 second: Optional[str] = None) -> List[str]:
    header = (f"{'method':<34}{'n':>5}{'cov':>6}" + (f"{'acc':>7}" if accuracy else "") +
              f"{'auroc':>8}{'auarc':>8}{'brier':>8}{'ece':>7}{'nll':>7}{'cal_nll':>9}"
              f"{'Δ vs ref':>10}{'95% CI':>17}" + (f"{'Δ vs best':>11}{'95% CI':>17}" if second else ""))
    lines = [header]
    for method, row in report.items():
        marker = " *" if method == reference else ""
        second_ci = row.get(f"delta_ci_vs_{second}") if second else None
        lines.append(
            f"{method + marker:<34}{row['n']:>5}{row['coverage']:>6.2f}"
            + (_cell(row.get("accuracy"), 7) if accuracy else "")
            + f"{_cell(row.get('auroc'), 8)}{_cell(row.get('auarc'), 8)}{_cell(row.get('brier'), 8)}"
            f"{_cell(row.get('ece'), 7)}{_cell(row.get('nll'), 7)}{_cell(row.get('cal_nll'), 9)}"
            f"{_cell(row.get('delta_auroc'), 10)}{_ci(row.get('delta_ci')):>17}"
            + (f"{_cell(row.get(f'delta_auroc_vs_{second}'), 11)}{_ci(second_ci):>17}" if second else ""))
    return lines


def _ci(interval) -> str:
    return f"[{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval else "—"


def _cell(value, width: int) -> str:
    return f"{'—':>{width}}" if value is None or value != value else f"{value:>{width}.3f}"
