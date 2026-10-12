"""Checks cells.pkl against the weighting study's consistency cells and against systems.csv for both Learned Weights rules."""

import pickle
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bayes  # noqa: E402
from bayes_study import _scores  # noqa: E402

SYSTEMS = HERE.parent / "per_system_results" / "systems.csv"
NAMES = {"weighted_sum": "weighted_sum_1se", "weighted_average": "weighted_average_1se"}


def replay(cell):
    warnings.filterwarnings("ignore")
    fit, ev = bayes.view(cell, "fit"), bayes.view(cell, "eval")
    return [{**{k: cell[k] for k in ("estimator", "setting", "task", "models")}, "method": method,
             **_scores.metrics(cell, *bayes.RULES[name](fit, ev))} for method, name in NAMES.items()]


def main():
    cells = pickle.loads((HERE / "cells.pkl").read_bytes())
    old = {(c["setting"].replace("_both", ""), c["task"], c["models"]): c
           for c in pickle.loads((HERE.parent / "weighting_study" / "cells.pkl").read_bytes())
           if c["setting"] in ("voting", "debate_r1_both", "debate_r2_both")}
    gaps = [max(np.abs(c[s] - old[c["setting"], c["task"], c["models"]][s]).max() for s in ("fit", "eval"))
            for c in cells if c["estimator"] == "cons"]
    print(f"consistency cells vs weighting_study/cells.pkl: largest confidence gap {max(gaps):.2e} over {len(gaps)} cells")
    with Pool() as pool:
        rows = pd.DataFrame([r for result in pool.map(replay, cells) for r in result])
    systems = pd.read_csv(SYSTEMS)
    joined = rows.merge(systems, on=["estimator", "setting", "task", "models", "method"], suffixes=("", "_saved"))
    for metric in ("p_brier", "auarc"):
        gap = (joined[metric] - joined[f"{metric}_saved"]).abs()
        print(f"{metric}: {len(joined)} rows matched, largest gap {gap.max():.2e}, rows above 1e-5: "
              f"{(gap > 1e-5).sum()}")
        print(joined.loc[gap > 1e-5, ["estimator", "setting", "task", "models", "method", metric, f"{metric}_saved"]]
              .to_string())


if __name__ == "__main__":
    main()
