"""Confidence report: test ranking metrics with bootstrap CIs, raw and validation-calibrated reliability."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from numpy import nan

from conf_compose.utils.calibration import CALIBRATORS
from conf_compose.utils.metrics import auarc, auroc, bootstrap_ci, brier, ece, nll

from .zero_shot import signal, signal_names


def confidence_report(validation: Sequence[Dict[str, Any]], test: Sequence[Dict[str, Any]],
                      calibrator: str = "beta", n_boot: int = 1000,
                      answered_only: bool = True) -> Dict[str, Dict[str, Any]]:
    if answered_only:
        validation = [record for record in validation if record["prediction"] is not None]
        test = [record for record in test if record["prediction"] is not None]
    test_correct = np.array([record["correct"] for record in test], dtype=float)
    val_correct = np.array([record["correct"] for record in validation], dtype=float)
    report: Dict[str, Dict[str, Any]] = {}
    can_calibrate = len(set(val_correct.tolist())) == 2
    if can_calibrate:
        report["constant"] = _reliability(np.full(len(test), val_correct.mean()), test_correct, "cal")
        report["constant"].update(auroc=float("nan"), auarc=float(test_correct.mean()), missing=0)

    for name in signal_names(test):
        scored = [record for record in test if record["confidence"][name] is not None]
        conf = np.array([record["confidence"][name] for record in scored])
        correct = np.array([record["correct"] for record in scored], dtype=float)
        row = {"missing": len(test) - len(scored), "coverage": len(scored) / len(test)}
        if len(scored) < 2 or len(set(correct.tolist())) < 2:
            report[name] = row
            continue
        row.update({"auroc": auroc(conf, correct), "auroc_ci": bootstrap_ci(auroc, conf, correct, n_boot),
                    "auarc": auarc(conf, correct), "auarc_ci": bootstrap_ci(auarc, conf, correct, n_boot)})
        row.update(_reliability(conf, correct, "raw"))
        if can_calibrate:
            fit = [record for record in validation if record["confidence"][name] is not None]
            if fit and len({record["correct"] for record in fit}) == 2:
                model = CALIBRATORS[calibrator]().fit([r["confidence"][name] for r in fit],
                                                      [r["correct"] for r in fit])
                row.update(_reliability(model.predict(conf), correct, "cal"))
        report[name] = row
    return report


def format_report(report: Dict[str, Dict[str, Any]]) -> List[str]:
    header = (f"{'signal':<30}{'auroc':>8}{'95% CI':>16}{'auarc':>8}{'raw_ece':>9}{'raw_brier':>10}"
              f"{'cal_ece':>9}{'cal_brier':>10}{'cal_nll':>9}{'coverage':>10}")
    lines = [header]
    for name, row in report.items():
        low, high = row.get("auroc_ci") or (nan, nan)
        ci = "—" if low != low else f"[{low:.3f}, {high:.3f}]"
        values = [_cell(row.get(key), width) for key, width in
                  (("auroc", 8), ("auarc", 8), ("raw_ece", 9), ("raw_brier", 10), ("cal_ece", 9), ("cal_brier", 10),
                   ("cal_nll", 9))]
        lines.append(f"{name:<30}{values[0]}{ci:>16}{''.join(values[1:])}{_cell(row.get('coverage'), 10)}")
    return lines


def _cell(value, width: int) -> str:
    return f"{'—':>{width}}" if value is None or value != value else f"{value:>{width}.3f}"


def _reliability(conf: np.ndarray, correct: np.ndarray, prefix: str) -> Dict[str, float]:
    return {f"{prefix}_ece": ece(conf, correct), f"{prefix}_brier": brier(conf, correct),
            f"{prefix}_nll": nll(conf, correct)}
