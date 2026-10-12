"""Does the log-odds sum lose on Brier because of its link? Equal-weight scores, every link fitted on the fitting split."""

import importlib.util
import pickle
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import isotonic_regression

HERE = Path(__file__).resolve().parent
WEIGHTING = HERE.parent / "weighting_study"
_spec = importlib.util.spec_from_file_location("weighting_methods", WEIGHTING / "methods.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_spec = importlib.util.spec_from_file_location("weighting_scores", WEIGHTING / "study.py")
_scores = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scores)
SETTINGS = ("voting", "debate_r1_both", "debate_r2_both")
KINDS = ("platt", "beta", "smooth", "isotonic")


def scores(p):
    """The two equal-weight pooled scores on the logit scale: log-odds sum and logit of the arithmetic mean."""
    return {"logodds": _base.logits(p).sum(1), "average": _base.logits(p.mean(1))}


def design(s, kind):
    """Platt is linear in the score; beta adds separate slopes for the score's positive and negative side."""
    if kind == "platt":
        return np.column_stack([np.ones(len(s)), s])
    return np.column_stack([np.ones(len(s)), -np.logaddexp(0, -s), np.logaddexp(0, s)])


SMOOTH_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, np.inf)


def tents(s, grid):
    """Piecewise-linear basis over the score: each column is one knot's tent."""
    s = np.clip(s, grid[0], grid[-1])
    return np.column_stack([np.interp(s, grid, np.eye(len(grid))[j]) for j in range(len(grid))])


def smooth(fit_s, fit_y, eval_s):
    """Platt plus a ridge-shrunk piecewise-linear bend at five quantile knots; cross-validation picks the ridge."""
    grid = np.unique(np.quantile(fit_s, np.linspace(0, 1, 5)))

    def fit(s, y, penalty):
        x = np.column_stack([np.ones(len(s)), s, tents(s, grid)])
        ridge = np.r_[0.0, 1e-6, np.full(len(grid), penalty if np.isfinite(penalty) else 1e12)]
        return _base.logistic(x, y, ridge)

    def linear(theta, s):
        return np.column_stack([np.ones(len(s)), s, tents(s, grid)]) @ theta
    theta = fit(fit_s, fit_y, _base.cross_validated(fit_s, fit_y, SMOOTH_GRID, fit, linear, one_se=True))
    return expit(linear(theta, fit_s)), expit(linear(theta, eval_s)), eval_s


def link(kind):
    def rule(fit_s, fit_y, eval_s):
        if kind == "isotonic":
            levels, inverse, counts = np.unique(fit_s, return_inverse=True, return_counts=True)
            means = np.bincount(inverse, fit_y) / counts
            step = np.clip(isotonic_regression(means, weights=counts).x, 1e-3, 1 - 1e-3)
            predict = lambda v: np.interp(v, levels, step)  # noqa: E731
            return predict(fit_s), predict(eval_s), eval_s
        if kind == "smooth":
            return smooth(fit_s, fit_y, eval_s)
        x = design(fit_s, kind)
        theta = _base.logistic(x, fit_y, np.r_[0.0, np.full(x.shape[1] - 1, 1e-3)])
        return expit(x @ theta), expit(design(eval_s, kind) @ theta), eval_s
    return rule


def score(cell):
    warnings.filterwarnings("ignore")
    fit, evaluation = scores(cell["fit"]), scores(cell["eval"])
    rows = []
    for pooled in fit:
        for kind in KINDS:
            result = _scores.metrics(cell, *link(kind)(fit[pooled], cell["fit_labels"], evaluation[pooled]))
            rows.append({**{k: cell[k] for k in ("setting", "task", "models")}, "method": f"{pooled}_{kind}", **result})
    return rows


def main():
    cells = [c for c in pickle.loads((WEIGHTING / "cells.pkl").read_bytes()) if c["setting"] in SETTINGS]
    with Pool() as pool:
        rows = [row for result in pool.imap_unordered(score, cells) for row in result]
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "links.csv", index=False)
    wide = frame.pivot_table(index=["setting", "task", "models"], columns="method", values="p_brier")
    for kind in KINDS:
        gap = 100 * (wide[f"logodds_{kind}"] - wide[f"average_{kind}"])
        cells_ = gap.groupby(level=["setting", "task"]).mean()
        print(f"\n{kind}: log-odds minus average p-Brier (negative = log-odds better); better in "
              f"{(cells_ < 0).sum()}/15 cells, {(gap < 0).sum()}/{len(gap)} groups")
        print(cells_.unstack("task").round(3).to_string())
    print("\nmean p-Brier x100 per method:")
    print((100 * frame.groupby("method").p_brier.mean()).round(3).to_string())


if __name__ == "__main__":
    main()
