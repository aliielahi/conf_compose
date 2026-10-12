"""Log-odds against probability pooling under two proper scoring rules (Brier and NLL), both after the table's Platt step."""

import importlib.util
import pickle
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from conf_compose.utils.calibration import PlattCalibrator

HERE = Path(__file__).resolve().parent
WEIGHTING = HERE.parent / "weighting_study"
_spec = importlib.util.spec_from_file_location("weighting_methods", WEIGHTING / "methods.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
SETTINGS = ("voting", "debate_r1_both", "debate_r2_both")
# (log-odds rule, probability rule) compared at equal weights and with learned weights.
PAIRS = {"equal": ("shared_scale", "mean"), "learned": ("weighted_sum_1se", "weighted_average_1se")}


def losses(cell, fit_conf, conf):
    fit_conf, conf = np.round(np.asarray(fit_conf, float), 12), np.round(np.asarray(conf, float), 12)
    q = np.clip(PlattCalibrator().fit(fit_conf, cell["fit_labels"]).predict(conf), 1e-6, 1 - 1e-6)
    y = cell["eval_labels"]
    return float(np.mean((q - y) ** 2)), float(-np.mean(y * np.log(q) + (1 - y) * np.log1p(-q)))


def score(cell):
    warnings.filterwarnings("ignore")
    rows = []
    for method in {m for pair in PAIRS.values() for m in pair}:
        values = cell["pipeline"][method] if method in cell["pipeline"] else \
            _base.RULES[method](cell["fit"], cell["fit_labels"], cell["eval"])
        brier, nll = losses(cell, values[0], values[1])
        rows.append({"setting": cell["setting"], "task": cell["task"], "models": cell["models"], "method": method,
                     "brier": brier, "nll": nll})
    return rows


def main():
    cells = [c for c in pickle.loads((WEIGHTING / "cells.pkl").read_bytes()) if c["setting"] in SETTINGS]
    with Pool() as pool:
        rows = [row for result in pool.imap_unordered(score, cells) for row in result]
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "scoring_rules.csv", index=False)
    wide = frame.pivot_table(index=["setting", "task", "models"], columns="method", values=["brier", "nll"])
    for name, (logodds, probability) in PAIRS.items():
        diff = pd.DataFrame({metric: 100 * (wide[metric][logodds] - wide[metric][probability])
                             for metric in ("brier", "nll")})
        cells_ = diff.groupby(level=["setting", "task"]).mean()
        print(f"\n{name} weights: {logodds} minus {probability}, x100 (negative = log-odds better)")
        print(cells_.unstack("setting").round(3).to_string())
        print(f"cells where log-odds is better: Brier {(cells_.brier < 0).sum()}/15, NLL {(cells_.nll < 0).sum()}/15; "
              f"groups: Brier {(diff.brier < 0).sum()}, NLL {(diff.nll < 0).sum()} of {len(diff)}")


if __name__ == "__main__":
    main()
