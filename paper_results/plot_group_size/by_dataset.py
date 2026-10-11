"""One figure per metric, three datasets in a row: gain against group size for voting and both debate rounds."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "paper_results/results"
# Debate pools each member's round-0 confidence with its confidence after the round, as in the main table.
SOURCES = {"voting": (RESULTS / "voting_protocol/atomic.csv", ""),
           "debate_r1": (RESULTS / "debate_protocol/run_40bca54388122738/cons/atomic.csv", "both_rounds:"),
           "debate_r2": (RESULTS / "debate_protocol/run_12a013aed30bd201/cons/atomic.csv", "both_rounds:")}
WEIGHTING = ROOT / "paper_results/weighting_study"
# The proposed learned log-odds weights come from the weighting study, scored on the same groups and questions.
WEIGHTED = {"voting": "voting", "debate_r1_both": "debate_r1", "debate_r2_both": "debate_r2"}
OUTPUTS = {"logodds_sum": "group_size", "weighted_sum": "group_size_weighted"}
SERIES = {"voting": ("Voting", "2A78D6", "o", "*"), "debate_r1": ("Debate, Round 1", "EB6834", "s", "square*"),
          "debate_r2": ("Debate, Round 2", "1BAF7A", "^", "triangle*")}
METRICS = {"delta_answer_matched_auarc": ("auarc", r"$\Delta$ Acc-Matched AUARC"),
           "delta_p_brier": ("brier", r"$\Delta$ p-Brier")}
TASKS = {"csqa": "CSQA", "gsm8k": "GSM8K", "truthfulqa": "TruthfulQA"}
XLABEL = "Number of Agents in Group"
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#eeede9"


def style():
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 6.5, "legend.fontsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "axes.edgecolor": MUTED, "axes.linewidth": 0.5,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.top": True, "ytick.right": True, "pdf.fonttype": 42,
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 600})


def weighted():
    """Proposed weights minus the best single agent, using the reference the pipeline's deltas imply."""
    sys.path.insert(0, str(WEIGHTING))
    from table import references
    scores = pd.read_csv(WEIGHTING / "scores.csv")
    scores = scores[scores.method == "weighted_sum_1se"].merge(references(), on=["setting", "task", "models"],
                                                             validate="one_to_one")
    return pd.DataFrame({"setting": scores.setting.map(WEIGHTED), "task": scores.task, "n_models": scores.n_models,
                         "delta_answer_matched_auarc": scores.auarc - scores.reference_auarc,
                         "delta_p_brier": scores.p_brier - scores.reference_p_brier}).dropna(subset=["setting"])


def load(method):
    if method == "weighted_sum":
        frame = weighted()
        return frame.assign(**{m: 100 * frame[m] for m in METRICS})
    frames = []
    for setting, (path, prefix) in SOURCES.items():
        frame = pd.read_csv(path)
        if "estimator" in frame:
            frame = frame[frame.estimator == "cons"]
        frames.append(frame[frame.method == prefix + method].assign(setting=setting))
    frame = pd.concat(frames)
    return frame[["setting", "task", "n_models", *METRICS]].assign(**{m: 100 * frame[m] for m in METRICS})


def series(frame, metric):
    """(group size, mean gain over groups) per dataset and setting, shared by both renderers."""
    return {(task, setting): list(frame[(frame.task == task) & (frame.setting == setting)]
                                  .groupby("n_models")[metric].mean().items())
            for task in TASKS for setting in SERIES}


def limits(points, task):
    low = min(min(0, y) for s in SERIES for _, y in points[task, s])
    high = max(max(0, y) for s in SERIES for _, y in points[task, s])
    pad = high - low
    return low - 0.08 * pad, high + 0.08 * pad


def figure(points, ylabel, sizes):
    fig, axes = plt.subplots(1, len(TASKS), figsize=(6.3, 1.85))
    for axis, (task, name) in zip(axes, TASKS.items()):
        axis.axhline(0, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
        axis.grid(axis="y", color=GRID, lw=0.4, zorder=0)
        for setting, (label, color, marker, _) in SERIES.items():
            x, y = zip(*points[task, setting])
            axis.plot(x, y, color=f"#{color}", marker=marker, lw=0.9, markersize=3, markeredgecolor="white",
                      markeredgewidth=0.4, label=label, zorder=2)
        axis.set_ylim(*limits(points, task))
        axis.set_xlim(sizes[0] - 0.4, sizes[-1] + 0.4)
        axis.set_xticks(sizes)
        axis.set_title(name, fontsize=6.8, color=INK, pad=3)
        axis.set_xlabel(XLABEL)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.12),
               handlelength=1.8, columnspacing=1.6)
    fig.subplots_adjust(wspace=0.28)
    return fig


def pgfplots(points, ylabel, sizes):
    """Compiled by LaTeX itself; needs pgfplots with the groupplots library. Meant for a figure* environment."""
    middle = len(TASKS) // 2
    lines = ["% Needs in the preamble: \\usepackage{pgfplots} \\usepgfplotslibrary{groupplots} \\pgfplotsset{compat=1.17}"]
    lines += [f"\\definecolor{{gs{setting.replace('_', '')}}}{{HTML}}{{{color}}}"
              for setting, (_, color, _, _) in SERIES.items()]
    lines += ["\\begin{tikzpicture}",
              f"\\begin{{groupplot}}[group style={{group size={len(TASKS)} by 1, horizontal sep=1.1cm}},",
              "  scale only axis, width=0.25\\textwidth, height=0.18\\textwidth,",
              f"  xmin={sizes[0] - 0.4}, xmax={sizes[-1] + 0.4}, xtick={{{','.join(map(str, sizes))}}},",
              "  axis line style={black!55, line width=0.4pt}, tick style={black!55, line width=0.4pt}, tick align=inside,",
              "  tick label style={font=\\tiny, black!70}, label style={font=\\scriptsize},",
              "  ymajorgrids, grid style={black!7, line width=0.3pt},",
              f"  xlabel={{{XLABEL}}}, title style={{font=\\footnotesize, yshift=-3pt}},",
              "  every axis plot/.append style={line width=0.7pt, mark size=1.4pt}]"]
    for index, (task, name) in enumerate(TASKS.items()):
        low, high = limits(points, task)
        options = [f"title={{{name}}}", f"ymin={low:.3f}", f"ymax={high:.3f}"]
        if index == 0:
            options.append(f"ylabel={{{ylabel}}}")
        if index == middle:
            options += ["legend columns=3",
                        "legend style={at={(0.5,1.16)}, anchor=south, draw=none, fill=none, font=\\small, /tikz/every even column/.append style={column sep=8pt}}"]
        lines.append(f"\\nextgroupplot[{', '.join(options)}]")
        lines.append(f"\\addplot[black!45, dashed, line width=0.4pt, forget plot] coordinates {{({sizes[0] - 0.4},0) ({sizes[-1] + 0.4},0)}};")
        for setting, (label, _, _, mark) in SERIES.items():
            coordinates = " ".join(f"({x},{y:.4f})" for x, y in points[task, setting])
            color = f"gs{setting.replace('_', '')}"
            size = ", mark size=2pt" if mark == "triangle*" else ""
            lines.append(f"\\addplot[color={color}, mark={mark}{size}, mark options={{draw=white, line width=0.25pt, fill={color}}}] "
                         f"coordinates {{{coordinates}}};")
            if index == middle:
                lines.append(f"\\addlegendentry{{{label}}}")
    lines += ["\\end{groupplot}", "\\end{tikzpicture}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=list(OUTPUTS), default="logodds_sum")
    method = parser.parse_args().method
    style()
    frame = load(method)
    sizes = sorted(int(size) for size in frame.n_models.unique())
    for metric, (name, ylabel) in METRICS.items():
        points = series(frame, metric)
        fig = figure(points, ylabel, sizes)
        stem = HERE / f"{OUTPUTS[method]}_{name}"
        for suffix in ("png", "pdf"):
            fig.savefig(stem.with_suffix(f".{suffix}"))
        plt.close(fig)
        stem.with_suffix(".tex").write_text(pgfplots(points, ylabel, sizes))
        print(f"wrote {stem}.png/.pdf/.tex")


if __name__ == "__main__":
    main()
