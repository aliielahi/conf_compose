"""Each Bayes variant minus the probability-space Learned Weights: wins, mean gaps and paired tests per estimator and
fold seed (bayes_summary.csv), per-setting x dataset mean gaps (bayes_cells.csv) and absolute means (bayes_absolute.csv),
all in table points (x100)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

HERE = Path(__file__).resolve().parent
BASE = "weighted_average_1se"
TASKS = ["csqa", "boolq", "gsm8k", "truthfulqa", "gpqa"]
KEYS = ["estimator", "seed", "setting", "task", "models"]


def tests(gap, better):
    """Wins (strictly better), ties, mean gap, Wilcoxon and sign-test p-values for one metric's per-group gaps."""
    gap = gap.to_numpy()
    tied = np.isclose(gap, 0, atol=1e-9)
    wins = int((better(gap) & ~tied).sum())
    losses = int((~better(gap) & ~tied).sum())
    return {"wins": wins, "ties": int(tied.sum()), "mean": gap.mean(),
            "wilcoxon_p": wilcoxon(gap[~tied]).pvalue if (~tied).sum() > 10 else np.nan,
            "sign_p": binomtest(wins, wins + losses).pvalue if wins + losses else np.nan}


def gaps(scores):
    wide = scores.pivot_table(index=KEYS, columns="method", values=["p_brier", "auarc"])
    frames = []
    for method in wide["p_brier"].columns.drop(BASE):
        frames.append(pd.DataFrame({"method": method, "brier": 100 * (wide["p_brier"][method] - wide["p_brier"][BASE]),
                                    "auarc": 100 * (wide["auarc"][method] - wide["auarc"][BASE])}).dropna())
    return pd.concat(frames).reset_index()


def summary(frame):
    rows = []
    for (method, estimator, seed), group in frame.groupby(["method", "estimator", "seed"]):
        brier, auarc = tests(group.brier, lambda g: g < 0), tests(group.auarc, lambda g: g > 0)
        rows.append({"method": method, "estimator": estimator, "seed": seed, "groups": len(group),
                     **{f"brier_{k}": v for k, v in brier.items()}, **{f"auarc_{k}": v for k, v in auarc.items()},
                     "both_wins": int(((group.brier < -1e-9) & (group.auarc > 1e-9)).sum())})
    return pd.DataFrame(rows)


def main():
    scores = pd.read_csv(HERE / "bayes_scores.csv")
    (100 * scores.groupby(["method", "estimator", "seed", "setting", "task"])[["p_brier", "auarc"]].mean()).reset_index() \
        .to_csv(HERE / "bayes_absolute.csv", index=False)
    frame = gaps(scores)
    table = summary(frame)
    table.to_csv(HERE / "bayes_summary.csv", index=False)
    cells = frame.groupby(["method", "estimator", "seed", "setting", "task"])[["brier", "auarc"]].mean().reset_index()
    cells.to_csv(HERE / "bayes_cells.csv", index=False)
    pd.set_option("display.width", 250)
    columns = ["groups", "brier_wins", "brier_mean", "brier_wilcoxon_p", "auarc_wins", "auarc_mean", "auarc_wilcoxon_p",
               "both_wins"]
    print("seed 0 (wins out of groups; means in points; negative Brier and positive AUARC favour the variant):")
    print(table[table.seed == 0].set_index(["estimator", "method"])[columns].round(4).to_string())
    print("\nacross fold seeds (min..max):")
    spread = table.groupby(["estimator", "method"]).agg(
        brier_wins=("brier_wins", lambda s: f"{s.min()}..{s.max()}"), brier_mean=("brier_mean", lambda s: f"{s.min():+.3f}..{s.max():+.3f}"),
        auarc_wins=("auarc_wins", lambda s: f"{s.min()}..{s.max()}"), auarc_mean=("auarc_mean", lambda s: f"{s.min():+.3f}..{s.max():+.3f}"))
    print(spread.to_string())
    for method in sys.argv[1:]:
        for estimator in ("cons", "seq_reasoned"):
            chosen = cells[(cells.method == method) & (cells.estimator == estimator) & (cells.seed == 0)]
            text = (chosen.brier.map("{:+.2f}".format) + " / " + chosen.auarc.map("{:+.2f}".format))
            print(f"\n{method}, {estimator}, seed 0: dBrier / dAUARC per setting x dataset")
            print(text.set_axis(pd.MultiIndex.from_frame(chosen[["setting", "task"]])).unstack("task")
                  .reindex(columns=TASKS).to_string())


if __name__ == "__main__":
    main()
