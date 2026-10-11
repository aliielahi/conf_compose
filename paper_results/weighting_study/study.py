"""Scores the pipeline's rules and the new ones per group with the table's metrics (p-Brier, AUARC) into scores.csv."""

import pickle
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from conf_compose.utils.calibration import PlattCalibrator
from conf_compose.utils.metrics import auarc, brier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from methods import RULES  # noqa: E402


def metrics(cell, fit_conf, conf, rank):
    fit_conf, conf, rank = (np.round(np.asarray(v, dtype=float), 12) for v in (fit_conf, conf, rank))
    fit_y, y = cell["fit_labels"], cell["eval_labels"]
    keep = ~np.isnan(fit_conf)
    if np.isnan(conf).any() or keep.sum() == 0:
        return None
    platt = PlattCalibrator().fit(fit_conf[keep], fit_y[keep]).predict(conf)
    return {"p_brier": float(brier(platt, y)), "auarc": float(auarc(rank, y))}


def score(cell):
    warnings.filterwarnings("ignore")
    outputs = {**cell["pipeline"], **{name: rule(cell["fit"], cell["fit_labels"], cell["eval"])
                                      for name, rule in RULES.items()}}
    rows = []
    for name, values in outputs.items():
        result = metrics(cell, *values)
        if result is not None:
            rows.append({**{k: cell[k] for k in ("setting", "task", "models", "n_models")}, "method": name,
                         "n_fit": len(cell["fit_labels"]), "n_eval": len(cell["eval_labels"]), **result})
    return rows


def main():
    cells = pickle.loads((HERE / "cells.pkl").read_bytes())
    with Pool() as pool:
        rows = [row for result in pool.imap_unordered(score, cells) for row in result]
    pd.DataFrame(rows).to_csv(HERE / "scores.csv", index=False)
    print(f"wrote {HERE / 'scores.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
