"""Group-size curves of consistency (solid) against one sequence-probability variant (dashed), per metric."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SETTINGS = {"voting": ("Voting", "#2A78D6", "o"), "debate_r1": ("Debate, Round 1", "#EB6834", "s"),
            "debate_r2": ("Debate, Round 2", "#1BAF7A", "^")}
VARIANTS = {"seq_label": "Seq. Prob. (label-normalized)", "seq_token": "Seq. Prob. (token-mean)",
            "seq_reasoned": "Seq. Prob. (after reasoning)"}
METRICS = {"delta_auarc": ("auarc", r"$\Delta$ Acc-Matched AUARC"), "delta_p_brier": ("brier", r"$\Delta$ p-Brier")}
# Output suffix per pooling rule: the plain log-odds sum and the proposed learned weights.
METHODS = {"logodds_sum": "", "weighted_sum": "_weighted"}
TASKS = {"csqa": "CSQA", "boolq": "BoolQ", "gsm8k": "GSM8K", "truthfulqa": "TruthfulQA", "gpqa": "GPQA"}
XLABEL = "Number of Agents in Group"
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#eeede9"


def style():
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 6.5, "legend.fontsize": 6.5, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "axes.edgecolor": MUTED, "axes.linewidth": 0.5,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.top": True, "ytick.right": True, "pdf.fonttype": 42,
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 600})


def draw(frame, variant, metric, ylabel):
    figure, axes = plt.subplots(1, len(TASKS), figsize=(8.4, 1.75))
    sizes = sorted(frame.n_models.unique())
    for axis, (task, name) in zip(axes, TASKS.items()):
        axis.axhline(0, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
        axis.grid(axis="y", color=GRID, lw=0.4, zorder=0)
        for estimator, dashes, fill in (("cons", "-", None), (variant, (0, (3, 1.6)), "white")):
            for setting, (_, color, marker) in SETTINGS.items():
                rows = frame[(frame.task == task) & (frame.estimator == estimator) & (frame.setting == setting)]
                if rows.empty:
                    continue
                means = 100 * rows.groupby("n_models")[metric].mean()
                axis.plot(means.index, means.values, color=color, ls=dashes, marker=marker, lw=0.9, markersize=3,
                          markerfacecolor=fill or color, markeredgecolor=color if fill else "white",
                          markeredgewidth=0.6 if fill else 0.4, zorder=2)
        axis.set_xlim(sizes[0] - 0.4, sizes[-1] + 0.4)
        axis.set_xticks(sizes)
        axis.set_title(name, fontsize=6.8, color=INK, pad=3)
        axis.set_xlabel(XLABEL)
    axes[0].set_ylabel(ylabel)
    handles = [Line2D([], [], color=color, marker=marker, lw=0.9, markersize=3, markeredgecolor="white",
                      markeredgewidth=0.4, label=f"Consistency, {label}") for label, color, marker in SETTINGS.values()]
    handles += [Line2D([], [], color=color, marker=marker, lw=0.9, ls=(0, (3, 1.6)), markersize=3,
                       markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.6,
                       label=f"{VARIANTS[variant]}, {label}") for label, color, marker in SETTINGS.values()]
    figure.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.26),
                  handlelength=2.4, columnspacing=1.4)
    figure.subplots_adjust(wspace=0.38)
    return figure


def main():
    style()
    frame = pd.read_csv(HERE / "deltas.csv")
    for method, suffix in METHODS.items():
        rows = frame[frame.method == method]
        for variant in VARIANTS:
            if variant not in set(rows.estimator):
                continue
            for metric, (name, ylabel) in METRICS.items():
                figure = draw(rows, variant, metric, ylabel)
                path = HERE / f"{variant}{suffix}_{name}.png"
                figure.savefig(path)
                plt.close(figure)
                print(f"wrote {path}")


if __name__ == "__main__":
    main()
