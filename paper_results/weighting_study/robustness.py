"""Re-runs the proposed rules with other fold seeds and fold counts, to check the zero-loss result is not luck."""

import pickle
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import methods  # noqa: E402
from study import metrics  # noqa: E402

CONFIGS = ((0, 5), (1, 5), (2, 5), (0, 10))
RULES = {"weighted_sum": ("logodds_sum", methods.LOGIT_GRID, methods.anchored_fit, methods.anchored_linear, True),
         "weighted_average": ("mean", methods.AVERAGE_GRID, methods.average_fit, methods.average_linear, False)}


def run(job):
    cell, seed, folds = job
    warnings.filterwarnings("ignore")
    methods.SEED, methods.FOLDS = seed, folds
    row = {k: cell[k] for k in ("setting", "task", "models")} | {"seed": seed, "folds": folds}
    for name, (base, grid, fit, linear, on_logits) in RULES.items():
        x = methods.logits(cell["fit"]) if on_logits else cell["fit"]
        row[f"{name}_fallback"] = bool(np.isinf(methods.cross_validated(x, cell["fit_labels"], grid, fit, linear, True)))
        rule = methods.weighted_sum(one_se=True) if on_logits else methods.weighted_average(one_se=True)
        new = metrics(cell, *rule(cell["fit"], cell["fit_labels"], cell["eval"]))
        old = metrics(cell, *cell["pipeline"][base])
        row[f"{name}_brier"] = 100 * (new["p_brier"] - old["p_brier"])
        row[f"{name}_auarc"] = 100 * (new["auarc"] - old["auarc"])
    return row


def main():
    cells = pickle.loads((HERE / "cells.pkl").read_bytes())
    with Pool() as pool:
        frame = pd.DataFrame(pool.map(run, [(c, s, f) for c in cells for s, f in CONFIGS]))
    frame.to_csv(HERE / "robustness.csv", index=False)
    for (seed, folds), group in frame.groupby(["seed", "folds"]):
        line = [f"seed {seed}, {folds} folds:"]
        for name in RULES:
            means = group.groupby(["setting", "task"])[[f"{name}_brier", f"{name}_auarc"]].mean()
            losses = int(((means[f"{name}_brier"] > 0.1) | (means[f"{name}_auarc"] < -0.1)).sum())
            line.append(f"{name} dBrier {means[f'{name}_brier'].mean():+.2f} dAUARC {means[f'{name}_auarc'].mean():+.2f} "
                        f"losses {losses}/25 equal-weight fallback {group[f'{name}_fallback'].mean():.0%};")
        print(" ".join(line))


if __name__ == "__main__":
    main()
