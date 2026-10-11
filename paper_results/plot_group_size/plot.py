"""Gains over the best single agent as the group grows from 2 to 6 agents, for voting and debate."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
VOTING = ROOT / "paper_results/results/voting_protocol/atomic.csv"
DEBATE = ROOT / "paper_results/results/debate_protocol/run_40bca54388122738/cons/atomic.csv"
DEBATE_SOLO = ROOT / "results/debate_composition/run_40bca54388122738/metrics.csv"
# Names as in the main table; debate pools both rounds, voting pools the one round there is.
METHODS = {"logodds_sum": "log-odds sum", "logistic_pool": "learned weights (log-odds)",
           "kahn_diagonal": "diagonal covariance", "kahn": "full covariance"}
TASKS = {"csqa": ("CSQA", "#2a78d6", "o"), "boolq": ("BoolQ", "#eb6834", "s"), "gsm8k": ("GSM8K", "#1baf7a", "^"),
         "truthfulqa": ("TruthfulQA", "#eda100", "D"), "gpqa": ("GPQA (low power)", "#e87ba4", "v")}
# About 100 evaluation questions and two groups at sizes 5 and 6: drawn faintly and kept out of the mean.
NOISY = {"gpqa"}
METRICS = {"accuracy": "$\\Delta$ accuracy $\\uparrow$", "auarc": "$\\Delta$ AUARC (same answer) $\\uparrow$",
           "brier": "$\\Delta$ p-Brier (lower is better)"}
PROTOCOLS = {"voting": "Voting", "debate": "Debate"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ebeae6"


def style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "savefig.bbox": "tight", "savefig.dpi": 300})


def load(method):
    """One row per protocol, dataset and group, in points; debate accuracy is against round 0's best agent."""
    voting = pd.read_csv(VOTING)
    voting = voting[(voting.estimator == "cons") & (voting.method == method)].assign(protocol="voting")
    voting["accuracy_gain"] = voting.accuracy - voting.reference_accuracy
    debate = pd.read_csv(DEBATE)
    debate = debate[debate.method == f"both_rounds:{method}"].assign(protocol="debate")
    solo = pd.read_csv(DEBATE_SOLO)
    solo = solo.loc[solo.method == "best_solo", ["task", "models", "accuracy_all"]]
    debate = debate.merge(solo, on=["task", "models"], how="left", validate="one_to_one")
    if debate.accuracy_all.isna().any():
        raise ValueError("a debate group has no round-0 best agent")
    debate["accuracy_gain"] = debate.accuracy - debate.accuracy_all
    frame = pd.concat([voting, debate])
    return pd.DataFrame({"protocol": frame.protocol, "task": frame.task, "size": frame.n_models,
                         "accuracy": 100 * frame.accuracy_gain,
                         "auarc": 100 * frame.delta_answer_matched_auarc, "brier": 100 * frame.delta_p_brier})


def draw(frame, method):
    sizes = sorted(frame["size"].unique())
    counts = frame[(frame.protocol == "voting") & (frame.task == "csqa")].groupby("size").size()
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 3.9), sharex=True)
    for row, (protocol, protocol_name) in enumerate(PROTOCOLS.items()):
        data = frame[frame.protocol == protocol]
        for column, (metric, metric_name) in enumerate(METRICS.items()):
            axis = axes[row, column]
            axis.axhline(0, color=MUTED, lw=0.8, zorder=1)
            axis.grid(axis="y", color=GRID, lw=0.6, zorder=0)
            for task, (name, color, marker) in TASKS.items():
                means = data[data.task == task].groupby("size")[metric].mean()
                noisy = task in NOISY
                axis.plot(means.index, means.values, color=color, marker=marker, lw=0.9 if noisy else 1.1,
                          ls=(0, (3, 2)) if noisy else "-", alpha=0.5 if noisy else 0.95, markersize=3.6,
                          markeredgecolor="white", markeredgewidth=0.5, zorder=2)
            # Each dataset weighs equally, whatever its size.
            steady = data[~data.task.isin(NOISY)]
            overall = steady.groupby(["size", "task"])[metric].mean().groupby("size").mean()
            axis.plot(overall.index, overall.values, color=INK, lw=2.2, zorder=3, solid_capstyle="round")
            if row == 0:
                axis.set_title(metric_name, color=INK)
            if column == 0:
                axis.set_ylabel(f"{protocol_name}\n(points)")
            if row == 1:
                axis.set_xlabel("agents per group")
            axis.set_xticks(sizes)
    handles = [plt.Line2D([], [], color=color, marker=marker, lw=1.1, markersize=4, markeredgecolor="white",
                          ls=(0, (3, 2)) if task in NOISY else "-", alpha=0.5 if task in NOISY else 1, label=name)
               for task, (name, color, marker) in TASKS.items()]
    handles.append(plt.Line2D([], [], color=INK, lw=2.2, label="mean, excl. GPQA"))
    figure.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.03),
                  handlelength=1.6, columnspacing=1.1)
    groups = ", ".join(str(counts[size]) for size in sizes)
    figure.text(0.5, -0.06, f"{groups} groups at sizes {sizes[0]}-{sizes[-1]}, the same groups on every dataset. "
                "Deltas against the best single agent (debate accuracy: its round-0 answer).\n"
                f"Method: {METHODS[method]} (debate: pooled over both rounds).", ha="center", fontsize=6.5,
                color=MUTED, linespacing=1.4)
    figure.subplots_adjust(wspace=0.32, hspace=0.18)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=list(METHODS), default="logodds_sum")
    method = parser.parse_args().method
    style()
    figure = draw(load(method), method)
    for suffix in ("pdf", "png"):
        figure.savefig(HERE / f"group_size_{method}.{suffix}")
    print(f"wrote {HERE / f'group_size_{method}'}.pdf/.png")


if __name__ == "__main__":
    main()
