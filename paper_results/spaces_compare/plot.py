"""Learned weights in log-odds space (solid) against probability space (dotted), against group size: p-Brier and
ΔAUARC on all five datasets, one figure per confidence estimator and metric, as PNG and pgfplots."""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SYSTEMS = HERE.parent / "per_system_results" / "systems.csv"
# The paper's pgfplots style, shared with the seq-vs-cons figures.
_spec = importlib.util.spec_from_file_location("paper_figures", HERE.parent / "seq_vs_cons" / "paper_figures.py")
paper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paper)

ESTIMATORS = {"cons": "consistency", "seq_reasoned": "seq_prob"}
# Pooling rule -> (legend label, drawn hollow and dotted).
RULES = {"weighted_sum": ("Log-odds Space", False), "weighted_average": ("Probability Space", True)}
METRICS = {"delta_auarc": ("auarc", r"$\Delta$ Acc-Matched AUARC", True),
           "p_brier": ("brier", r"p-Brier ($\times 100$)", False)}
TASKS = paper.FIVE
MARKERS = {"voting": "o", "debate_r1": "s", "debate_r2": "^"}
COLORS = {setting: (color, MARKERS[setting]) for setting, (_, color, _) in paper.SETTINGS.items()}
DOTTED = (0, (1.2, 1.2))
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#eeede9"


def load():
    """One row per group, setting and rule, with the rule in the column the shared helpers group by."""
    frame = pd.read_csv(SYSTEMS)
    return frame.assign(delta_auarc=frame.auarc - frame.reference_auarc, confidence=frame.estimator,
                        estimator=frame.method)


def style():
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 6.5, "legend.fontsize": 6.5, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "axes.edgecolor": MUTED, "axes.linewidth": 0.5,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.top": True, "ytick.right": True, "pdf.fonttype": 42,
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 600})


def draw(series, ylabel, zero):
    sizes = sorted({x for values in series.values() for x, _ in values})
    figure, axes = plt.subplots(1, len(TASKS), figsize=(8.4, 1.75))
    for axis, (task, name) in zip(axes, TASKS.items()):
        if zero:
            axis.axhline(0, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
        axis.grid(axis="y", color=GRID, lw=0.4, zorder=0)
        for rule, (_, hollow) in RULES.items():
            for setting, (color, marker) in COLORS.items():
                x, y = zip(*series[task, rule, setting])
                axis.plot(x, y, color=f"#{color}", ls=DOTTED if hollow else "-", marker=marker, lw=0.9, markersize=3,
                          markerfacecolor="white" if hollow else f"#{color}",
                          markeredgecolor=f"#{color}" if hollow else "white",
                          markeredgewidth=0.6 if hollow else 0.4, zorder=2)
        axis.set_xlim(sizes[0] - 0.4, sizes[-1] + 0.4)
        axis.set_xticks(sizes)
        axis.set_title(name, fontsize=6.8, color=INK, pad=3)
        axis.set_xlabel("Number of Agents in Group")
    axes[0].set_ylabel(ylabel)
    handles = [Line2D([], [], color=INK, lw=0.9, ls=DOTTED if hollow else "-", label=label)
               for label, hollow in RULES.values()]
    handles += [Line2D([], [], color=f"#{color}", marker=marker, lw=0, markersize=3.5, markeredgecolor="white",
                       markeredgewidth=0.4, label=paper.SETTINGS[setting][0])
                for setting, (color, marker) in COLORS.items()]
    figure.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.1),
                  handlelength=2.0, columnspacing=1.3, handletextpad=0.5)
    figure.subplots_adjust(wspace=0.38)
    return figure


def main():
    style()
    frame = load()
    for confidence, stem in ESTIMATORS.items():
        rows = frame[frame.confidence == confidence]
        for metric, (name, ylabel, zero) in METRICS.items():
            series = paper.points(rows, TASKS, list(RULES), metric)
            path = HERE / f"{stem}_{name}"
            figure = draw(series, ylabel, zero)
            figure.savefig(path.with_suffix(".png"))
            plt.close(figure)
            tex = paper.figure(series, TASKS, [(rule, hollow) for rule, (_, hollow) in RULES.items()], ylabel, 0.143,
                               "0.7cm", line_legend=True, zero=zero,
                               line_labels=tuple(label for label, _ in RULES.values()))
            path.with_suffix(".tex").write_text(tex)
            print(f"wrote {path}.png/.tex")


if __name__ == "__main__":
    main()
