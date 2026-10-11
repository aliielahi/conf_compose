"""Each learned rule minus its equal-weight baseline, per setting and dataset, written to summary.md."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
COMPARISONS = {
    "logodds_sum": ["logistic_pool", "kahn_diagonal", "kahn", "weighted_sum", "weighted_sum_1se",
                    "weighted_sum_nonneg", "weighted_sum_nonneg_1se"],
    "mean": ["blp", "weighted_average", "weighted_average_1se"],
}
# A loss is a dataset mean worse than the baseline by more than this many points.
TOLERANCE = 0.1


def differences(scores, base, method):
    wide = scores.pivot_table(index=["setting", "task", "models"], columns="method", values=["p_brier", "auarc"])
    frame = pd.DataFrame({"brier": 100 * (wide["p_brier"][method] - wide["p_brier"][base]),
                          "auarc": 100 * (wide["auarc"][method] - wide["auarc"][base])})
    return frame.groupby(level=["setting", "task"]).mean()


def main():
    scores = pd.read_csv(HERE / "scores.csv")
    lines = ["# Learned weights minus equal weights", "",
             f"Points x100, mean over groups. p-Brier: negative is better. AUARC: positive is better. "
             f"Losses: dataset means worse than the baseline by more than {TOLERANCE} on either metric (of 25).", ""]
    for base, methods in COMPARISONS.items():
        lines += [f"## Against `{base}`", "", "| method | mean dBrier | mean dAUARC | worst dBrier | worst dAUARC | losses |",
                  "|---|---|---|---|---|---|"]
        cells = {}
        for method in methods:
            frame = differences(scores, base, method)
            cells[method] = frame
            losses = int(((frame.brier > TOLERANCE) | (frame.auarc < -TOLERANCE)).sum())
            lines.append(f"| {method} | {frame.brier.mean():+.2f} | {frame.auarc.mean():+.2f} | "
                         f"{frame.brier.max():+.2f} | {frame.auarc.min():+.2f} | {losses} |")
        table = pd.concat({m: f.brier.map("{:+.2f}".format) + " / " + f.auarc.map("{:+.2f}".format)
                           for m, f in cells.items()}, axis=1)
        lines += ["", "dBrier / dAUARC per setting and dataset:", "", "```\n" + table.to_string() + "\n```", ""]
    (HERE / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
