"""Mechanism checks behind findings.md: prior absorption, evidence saturation under the clip, and answer count vs accuracy."""

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bayes  # noqa: E402
from conf_compose.utils.metrics import auarc  # noqa: E402


def absorption(cells):
    """Largest gap between Learned Weights with and without a constant prior (per-agent or base rate)."""
    gaps = {"prior_base_rate": 0.0, "prior_agent_rate": 0.0}
    for cell in cells:
        fit, ev = bayes.view(cell, "fit"), bayes.view(cell, "eval")
        reference = bayes.RULES["weighted_sum_1se"](fit, ev)[1]
        for name in gaps:
            gaps[name] = max(gaps[name], np.abs(bayes.RULES[name](fit, ev)[1] - reference).max())
    return gaps


def saturation(cells):
    """Per estimator: members at the clip, questions tied at the top, and how the tie is broken."""
    rows = []
    for cell in cells:
        ev = bayes.view(cell, "eval")
        p, y = ev["p"], ev["y"]
        top = (p >= 1 - bayes.EPSILON).all(1)
        rows.append({"estimator": cell["estimator"], "task": cell["task"],
                     "members_clipped": float(((p >= 1 - bayes.EPSILON) | (p <= bayes.EPSILON)).mean()),
                     "questions_all_high": float(top.mean()), "accuracy_all_high": float(y[top].mean()) if top.any() else np.nan,
                     "accuracy_rest": float(y[~top].mean()),
                     "auarc_clipped_sum": 100 * auarc(np.round(bayes.evidence(p).sum(1), 12), y),
                     "auarc_exact_sum": 100 * auarc(np.round(ev["exact"].sum(1), 12), y),
                     "auarc_mean": 100 * auarc(np.round(p.mean(1), 12), y),
                     "spearman_sum_vs_mean": spearmanr(bayes.evidence(p).sum(1), p.mean(1)).statistic})
    return pd.DataFrame(rows)


def answer_count(cells):
    """Accuracy of the selected answer by the number of answers K, where K varies within a dataset."""
    rows = []
    for cell in cells:
        if cell["task"] not in ("truthfulqa", "gsm8k"):
            continue
        for split in ("fit", "eval"):
            data = bayes.view(cell, split)
            bins = np.minimum(data["k"], 6).astype(int)
            for k in np.unique(bins):
                rows.append({"estimator": cell["estimator"], "task": cell["task"], "k": k,
                             "n": int((bins == k).sum()), "correct": float(data["y"][bins == k].sum())})
    frame = pd.DataFrame(rows).groupby(["estimator", "task", "k"])[["n", "correct"]].sum()
    frame["accuracy"] = frame.correct / frame.n
    return frame.drop(columns="correct")


def main():
    warnings.filterwarnings("ignore")
    cells = pickle.loads((HERE / "cells.pkl").read_bytes())
    print("largest |confidence gap| vs weighted_sum_1se:", absorption(cells))
    pd.set_option("display.width", 200)
    print(saturation(cells).groupby(["estimator", "task"]).mean(numeric_only=True).round(3).to_string())
    print(answer_count(cells).round(3).to_string())


if __name__ == "__main__":
    main()
