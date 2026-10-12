"""Scores the Bayes-rule variants and both Learned Weights rows per group, estimator and fold seed into bayes_scores.csv."""

import importlib.util
import pickle
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bayes  # noqa: E402

_spec = importlib.util.spec_from_file_location("weighting_scores", HERE.parent / "weighting_study" / "study.py")
_scores = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scores)
SEEDS = (0, 1, 2, 3)


def score(job):
    """Every rule on one cell with one cross-validation fold seed."""
    warnings.filterwarnings("ignore")
    cell, seed, names = job
    bayes.set_seed(seed)
    fit, ev = bayes.view(cell, "fit"), bayes.view(cell, "eval")
    rows = []
    for name in names:
        result = _scores.metrics(cell, *bayes.RULES[name](fit, ev))
        if result is not None:
            rows.append({**{k: cell[k] for k in ("estimator", "setting", "task", "models", "n_models")},
                         "method": name, "seed": seed, **result})
    return rows


def main():
    names = sys.argv[1:] or list(bayes.RULES)
    cells = pickle.loads((HERE / "cells.pkl").read_bytes())
    jobs = [(cell, seed, names) for seed in SEEDS for cell in cells]
    with Pool() as pool:
        rows = [row for result in pool.imap_unordered(score, jobs, chunksize=4) for row in result]
    frame = pd.DataFrame(rows)
    path = HERE / "bayes_scores.csv"
    if path.exists() and sys.argv[1:]:
        old = pd.read_csv(path)
        frame = pd.concat([old[~old.method.isin(names)], frame], ignore_index=True)
    frame.to_csv(path, index=False)
    print(f"wrote {path} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
