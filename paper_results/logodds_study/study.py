"""Scores the candidate log-odds rules per group on the main-table settings into scores.csv, with the table's metrics."""

import importlib.util
import pickle
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
WEIGHTING = HERE.parent / "weighting_study"
sys.path.insert(0, str(HERE))

import rules  # noqa: E402

# The weighting study's scorer, loaded by path because this file shares its name.
_spec = importlib.util.spec_from_file_location("weighting_scores", WEIGHTING / "study.py")
_scores = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scores)
metrics = _scores.metrics

# The settings behind the main table's voting, debate round 1 and debate round 2 blocks.
SETTINGS = ("voting", "debate_r1_both", "debate_r2_both")


def candidates(cell):
    found = {"evidence_sum": rules.evidence_sum(), "evidence_sum_cv": rules.evidence_sum(one_se=False),
             "alpha_sum": rules.alpha_sum(), "evidence_weights": rules.evidence_weights(),
             "dispersion_sum": rules.dispersion_sum(), "dispersion_sum_cv": rules.dispersion_sum(one_se=False),
             "dispersion_weights": rules.dispersion_sum(agents=True),
             "gap_sum": rules.gap_sum(), "gap_sum_cv": rules.gap_sum(one_se=False),
             "gap_weights": rules.gap_sum(agents=True)}
    if cell["setting"].endswith("_both"):
        found["round_sum"] = rules.round_sum(cell["n_models"])
    return found


def score(cell):
    warnings.filterwarnings("ignore")
    rows = []
    for name, rule in candidates(cell).items():
        result = metrics(cell, *rule(cell["fit"], cell["fit_labels"], cell["eval"]))
        if result is not None:
            rows.append({**{k: cell[k] for k in ("setting", "task", "models", "n_models")}, "method": name, **result})
    return rows


def main():
    cells = [c for c in pickle.loads((WEIGHTING / "cells.pkl").read_bytes()) if c["setting"] in SETTINGS]
    with Pool() as pool:
        rows = [row for result in pool.imap_unordered(score, cells) for row in result]
    baselines = pd.read_csv(WEIGHTING / "scores.csv")
    baselines = baselines[baselines.setting.isin(SETTINGS)][["setting", "task", "models", "n_models", "method",
                                                             "p_brier", "auarc"]]
    pd.concat([baselines, pd.DataFrame(rows)]).to_csv(HERE / "scores.csv", index=False)
    print(f"wrote {HERE / 'scores.csv'}")


if __name__ == "__main__":
    main()
