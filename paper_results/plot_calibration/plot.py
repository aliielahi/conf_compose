"""Reliability of confidence in the system's answer: best single agent, raw log-odds sum, and its two corrections."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from conf_compose.utils.metrics import ece

HERE = Path(__file__).resolve().parent
TASKS = {"csqa": "CSQA", "boolq": "BoolQ", "gsm8k": "GSM8K", "truthfulqa": "TruthfulQA", "gpqa": "GPQA"}
PROTOCOLS = {"voting": "Voting", "debate": "Debate"}
METHODS = {
    "single": ("Best single agent", "#8a8985", "o"),
    "logodds_sum": ("Log-odds sum", "#e34948", "v"),
    "shared_rho": ("+ dependence correction ($\\rho$)", "#eda100", "D"),
    "shared_scale": ("+ fitted scale", "#2a78d6", "s"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ebeae6"


def style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "savefig.bbox": "tight", "savefig.dpi": 300})


def reliability(confidence, correct, bins=10):
    """Equal-mass bins spread the points over the range; ties at 1.0 can merge bins."""
    edges = np.unique(np.quantile(confidence, np.linspace(0, 1, bins + 1)))
    index = np.clip(np.searchsorted(edges, confidence, side="right") - 1, 0, len(edges) - 2)
    return np.array([(confidence[index == b].mean(), correct[index == b].mean()) for b in np.unique(index)])


def scores(frame):
    """ECE with the project's 10 equal-width bins, so the numbers match the tables."""
    return {m: 100 * ece(frame[frame.method == m].confidence.to_numpy(), frame[frame.method == m].correct.to_numpy())
            for m in METHODS}


def panel(axis, frame):
    axis.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    for method, (_, color, marker) in METHODS.items():
        rows = frame[frame.method == method]
        points = reliability(rows.confidence.to_numpy(), rows.correct.to_numpy())
        axis.plot(points[:, 0], points[:, 1], color=color, lw=1.2, alpha=0.9, zorder=2, clip_on=False)
        axis.scatter(points[:, 0], points[:, 1], s=16, color=color, marker=marker, edgecolor="white",
                     linewidth=0.6, zorder=3, clip_on=False)
    # ECE in a 2x2 block above the data: no panel has points that are both unconfident and accurate.
    for slot, (method, value) in enumerate(scores(frame).items()):
        x, y = 0.05 + 0.36 * (slot % 2), 0.95 - 0.1 * (slot // 2)
        _, color, marker = METHODS[method]
        axis.scatter(x, y, s=12, color=color, marker=marker, transform=axis.transAxes, clip_on=False, lw=0)
        axis.text(x + 0.04, y, f"{value:.1f}", transform=axis.transAxes, va="center", fontsize=7, color=INK)
    axis.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    axis.set_xticks((0, 0.5, 1), ("0", ".5", "1"))
    axis.set_yticks((0, 0.5, 1), ("0", ".5", "1"))


def legend(figure, columns, anchor):
    handles = [plt.Line2D([], [], color=color, marker=marker, lw=1.2, markersize=5, markeredgecolor="white",
                          label=label) for label, color, marker in METHODS.values()]
    figure.legend(handles=handles, loc="upper center", ncol=columns, frameon=False, bbox_to_anchor=(0.5, anchor),
                  handlelength=1.8, columnspacing=1.4)


def grid(data):
    """Datasets stay separate: pooling them lets opposite miscalibrations cancel and hides them."""
    figure, axes = plt.subplots(2, 5, figsize=(7.2, 3.45), sharex=True, sharey=True)
    for row, (protocol, protocol_name) in enumerate(PROTOCOLS.items()):
        for column, (task, task_name) in enumerate(TASKS.items()):
            axis = axes[row, column]
            panel(axis, data[(data.protocol == protocol) & (data.task == task)])
            if row == 0:
                axis.set_title(task_name, color=INK, pad=4)
            if column == 0:
                axis.set_ylabel(f"{protocol_name}\naccuracy")
            if row == 1:
                axis.set_xlabel("confidence")
    legend(figure, 4, 1.03)
    figure.text(0.5, -0.04, "Points: equal-mass bins. Numbers: ECE (points, 10 equal-width bins), keyed by marker.",
                ha="center", fontsize=7, color=MUTED)
    figure.subplots_adjust(wspace=0.12, hspace=0.16)
    return figure


def summary(data):
    """ECE per dataset; each method sits on its own row offset so no marker hides another."""
    figure, axes = plt.subplots(1, 2, figsize=(4.8, 2.3), sharey=True, sharex=True)
    tasks = list(TASKS)
    values = {(p, t): scores(data[(data.protocol == p) & (data.task == t)]) for p in PROTOCOLS for t in tasks}
    offsets = dict(zip(METHODS, np.linspace(-0.24, 0.24, len(METHODS))))
    for axis, (protocol, protocol_name) in zip(axes, PROTOCOLS.items()):
        for position, task in enumerate(tasks):
            cell = values[protocol, task]
            axis.plot([min(cell.values()), max(cell.values())], [position, position], color=GRID, lw=5,
                      zorder=1, solid_capstyle="round")
            for method, (_, color, marker) in METHODS.items():
                axis.scatter(cell[method], position + offsets[method], s=24, color=color, marker=marker,
                             edgecolor="white", linewidth=0.6, zorder=3)
        axis.set_title(protocol_name, color=INK)
        axis.set_xlabel("ECE (points, lower is better)")
        axis.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    axes[0].set_xlim(0, 1.05 * max(max(cell.values()) for cell in values.values()))
    axes[0].set_yticks(range(len(tasks)), [TASKS[t] for t in tasks])
    axes[0].invert_yaxis()
    legend(figure, 2, 1.06)
    figure.subplots_adjust(wspace=0.08, top=0.74)
    return figure


def main():
    style()
    data = pd.read_csv(HERE / "confidences.csv.gz")
    for name, draw in (("calibration_grid", grid), ("calibration_summary", summary)):
        figure = draw(data)
        for suffix in ("pdf", "png"):
            figure.savefig(HERE / f"{name}.{suffix}")
        plt.close(figure)
        print(f"wrote {HERE / name}.pdf/.png")


if __name__ == "__main__":
    main()
