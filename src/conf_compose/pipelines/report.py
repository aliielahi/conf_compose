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
    if len(validation):
        report["constant"] = _reliability(np.full(len(test), val_correct.mean()), test_correct, "cal")
        report["constant"].update(auroc=float("nan"), auarc=float(test_correct.mean()), missing=0)

    for name in signal_names(test):
        conf = np.array(signal(test, name))
        row = {
            "auroc": auroc(conf, test_correct),
            "auroc_ci": bootstrap_ci(auroc, conf, test_correct, n_boot),
            "auarc": auarc(conf, test_correct),
            "auarc_ci": bootstrap_ci(auarc, conf, test_correct, n_boot),
            "missing": sum(record["confidence"][name] is None for record in test),
        }
        row.update(_reliability(conf, test_correct, "raw"))
        if len(validation):
            calibrated = CALIBRATORS[calibrator]().fit(signal(validation, name), val_correct).predict(conf)
            row.update(_reliability(calibrated, test_correct, "cal"))
        report[name] = row
    return report


def format_report(report: Dict[str, Dict[str, Any]]) -> List[str]:
    header = (f"{'signal':<30}{'auroc':>8}{'95% CI':>16}{'auarc':>8}{'raw_ece':>9}{'raw_brier':>10}"
              f"{'cal_ece':>9}{'cal_brier':>10}{'cal_nll':>9}{'missing':>8}")
    lines = [header]
    for name, row in report.items():
        low, high = row.get("auroc_ci", (nan, nan))
        lines.append(f"{name:<30}{row['auroc']:>8.3f}{f'[{low:.3f}, {high:.3f}]':>16}{row['auarc']:>8.3f}"
                     f"{row.get('raw_ece', nan):>9.3f}{row.get('raw_brier', nan):>10.3f}"
                     f"{row.get('cal_ece', nan):>9.3f}{row.get('cal_brier', nan):>10.3f}"
                     f"{row.get('cal_nll', nan):>9.3f}{row['missing']:>8}")
    return lines


def _reliability(conf: np.ndarray, correct: np.ndarray, prefix: str) -> Dict[str, float]:
    return {f"{prefix}_ece": ece(conf, correct), f"{prefix}_brier": brier(conf, correct),
            f"{prefix}_nll": nll(conf, correct)}
