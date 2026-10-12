"""Each candidate minus the probability-space learned weights, per setting and dataset, in table points (x100)."""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = "weighted_average_1se"
CANDIDATES = sys.argv[1:] or ["weighted_sum_1se", "evidence_sum", "evidence_sum_cv", "alpha_sum", "round_sum",
                              "evidence_weights"]
TASKS = ["csqa", "boolq", "gsm8k", "truthfulqa", "gpqa"]


def main():
    scores = pd.read_csv(HERE / "scores.csv")
    wide = scores.pivot_table(index=["setting", "task", "models"], columns="method", values=["p_brier", "auarc"])
    for method in CANDIDATES:
        brier = 100 * (wide["p_brier"][method] - wide["p_brier"][BASE])
        auarc = 100 * (wide["auarc"][method] - wide["auarc"][BASE])
        cells = pd.DataFrame({"brier": brier, "auarc": auarc}).dropna().groupby(level=["setting", "task"]).mean()
        if cells.empty:
            continue
        both = int(((cells.brier < 0) & (cells.auarc > 0)).sum())
        print(f"\n{method}: Brier better in {(cells.brier < 0).sum()}/{len(cells)}, AUARC better in "
              f"{(cells.auarc > 0).sum()}/{len(cells)}, both in {both}; mean dBrier {cells.brier.mean():+.3f}, "
              f"mean dAUARC {cells.auarc.mean():+.3f}")
        table = (cells.brier.map("{:+.2f}".format) + "/" + cells.auarc.map("{:+.2f}".format)).unstack("task")
        print(table.reindex(columns=TASKS).to_string())


if __name__ == "__main__":
    main()
