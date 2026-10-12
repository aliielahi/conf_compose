"""Group-size curves of consistency (solid) against one sequence-probability variant (dotted), per metric."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SETTINGS = {"voting": ("Before Interaction", "2A78D6", "o", "*"),
            "debate_r1": ("Interaction Round 1", "EB6834", "s", "square*"),
            "debate_r2": ("Interaction Round 2", "1BAF7A", "^", "triangle*")}
VARIANTS = {"seq_label": "Seq. Prob. (Label-Normalized)", "seq_token": "Seq. Prob. (Token-Mean)",
            "seq_reasoned": "Seq. Prob. (After Reasoning)"}
METRICS = {"delta_accuracy": ("accuracy", r"$\Delta$ Accuracy"),
           "delta_auarc": ("auarc", r"$\Delta$ Acc-Matched AUARC"),
           "delta_p_brier": ("brier", r"$\Delta$ p-Brier"),
           "p_brier": ("brier_absolute", "p-Brier")}
# Only these metrics also get a pgfplots version.
TEX_METRICS = ("delta_auarc",)
# Output suffix per pooling rule: the plain log-odds sum and the proposed learned weights.
METHODS = {"logodds_sum": "", "weighted_sum": "_weighted"}
TASKS = {"csqa": "CSQA", "truthfulqa": "TruthfulQA", "gsm8k": "GSM8K"}
XLABEL = "Number of Agents in Group"
DOTTED = (0, (1.2, 1.2))
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#eeede9"


def style():
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 6.5, "legend.fontsize": 6.5, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "axes.edgecolor": MUTED, "axes.linewidth": 0.5,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.top": True, "ytick.right": True, "pdf.fonttype": 42,
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 600})


def series(frame, variant, metric):
    """(group size, mean over groups) per dataset, estimator and setting, shared by both renderers."""
    points = {}
    for task in TASKS:
        for estimator in ("cons", variant):
            for setting in SETTINGS:
                rows = frame[(frame.task == task) & (frame.estimator == estimator) & (frame.setting == setting)]
                points[task, estimator, setting] = list((100 * rows.groupby("n_models")[metric].mean()).items())
    return points


def draw(points, variant, ylabel, sizes, zero):
    figure, axes = plt.subplots(1, len(TASKS), figsize=(6.3, 1.85))
    for axis, (task, name) in zip(axes, TASKS.items()):
        if zero:
            axis.axhline(0, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
        axis.grid(axis="y", color=GRID, lw=0.4, zorder=0)
        for estimator, dashes, fill in (("cons", "-", None), (variant, DOTTED, "white")):
            for setting, (_, color, marker, _) in SETTINGS.items():
                if not points[task, estimator, setting]:
                    continue
                x, y = zip(*points[task, estimator, setting])
                axis.plot(x, y, color=f"#{color}", ls=dashes, marker=marker, lw=0.9, markersize=3,
                          markerfacecolor=fill or f"#{color}", markeredgecolor=f"#{color}" if fill else "white",
                          markeredgewidth=0.6 if fill else 0.4, zorder=2)
        axis.set_xlim(sizes[0] - 0.4, sizes[-1] + 0.4)
        axis.set_xticks(sizes)
        axis.set_title(name, fontsize=6.8, color=INK, pad=3)
        axis.set_xlabel(XLABEL)
    axes[0].set_ylabel(ylabel)
    handles = [Line2D([], [], color=INK, lw=0.9, label="Consistency"),
               Line2D([], [], color=INK, lw=0.9, ls=DOTTED, label=VARIANTS[variant])]
    handles += [Line2D([], [], color=f"#{color}", marker=marker, lw=0, markersize=3.5, markeredgecolor="white",
                       markeredgewidth=0.4, label=label) for label, color, marker, _ in SETTINGS.values()]
    figure.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.1),
                  handlelength=2.0, columnspacing=1.3, handletextpad=0.5)
    figure.subplots_adjust(wspace=0.28)
    return figure


def limits(points, task, variant):
    values = [y for e in ("cons", variant) for s in SETTINGS for _, y in points[task, e, s]]
    low, high = min(0, *values), max(0, *values)
    pad = high - low
    return low - 0.08 * pad, high + 0.08 * pad


def pgfplots(points, variant, ylabel, sizes):
    """Compiled by LaTeX itself; needs pgfplots with the groupplots library. Meant for a figure* environment."""
    middle = len(TASKS) // 2
    lines = ["% Needs in the preamble: \\usepackage{pgfplots} \\usepgfplotslibrary{groupplots} \\pgfplotsset{compat=1.17}"]
    lines += [f"\\definecolor{{sc{setting.replace('_', '')}}}{{HTML}}{{{color}}}"
              for setting, (_, color, _, _) in SETTINGS.items()]
    lines += ["\\begin{tikzpicture}",
              f"\\begin{{groupplot}}[group style={{group size={len(TASKS)} by 1, horizontal sep=1.1cm}},",
              "  scale only axis, width=0.25\\textwidth, height=0.18\\textwidth,",
              f"  xmin={sizes[0] - 0.4}, xmax={sizes[-1] + 0.4}, xtick={{{','.join(map(str, sizes))}}},",
              "  axis line style={black!55, line width=0.4pt}, tick style={black!55, line width=0.4pt}, tick align=inside,",
              "  tick label style={font=\\tiny, black!70}, label style={font=\\scriptsize},",
              "  ymajorgrids, grid style={black!7, line width=0.3pt},",
              f"  xlabel={{{XLABEL}}}, title style={{font=\\footnotesize, yshift=-3pt, text height=1.6ex, text depth=0.4ex}},",
              "  every axis plot/.append style={line width=0.7pt, mark size=1.4pt}]"]
    for index, (task, name) in enumerate(TASKS.items()):
        low, high = limits(points, task, variant)
        options = [f"title={{{name}}}", f"ymin={low:.3f}", f"ymax={high:.3f}"]
        if index == 0:
            options.append(f"ylabel={{{ylabel}}}")
        if index == middle:
            options += ["legend columns=5",
                        "legend style={at={(0.5,1.3)}, anchor=south, draw=none, fill=none, font=\\scriptsize, "
                        "/tikz/every even column/.append style={column sep=5pt}}"]
        lines.append(f"\\nextgroupplot[{', '.join(options)}]")
        lines.append(f"\\addplot[black!45, dashed, line width=0.4pt, forget plot] coordinates "
                     f"{{({sizes[0] - 0.4},0) ({sizes[-1] + 0.4},0)}};")
        if index == middle:
            lines += ["\\addlegendimage{black!80, line width=0.7pt}", "\\addlegendentry{Consistency}",
                      "\\addlegendimage{black!80, line width=0.7pt, densely dotted}",
                      f"\\addlegendentry{{{VARIANTS[variant]}}}"]
            for setting, (label, _, _, mark) in SETTINGS.items():
                color = f"sc{setting.replace('_', '')}"
                lines += [f"\\addlegendimage{{only marks, mark={mark}, color={color}, mark size=1.8pt}}",
                          f"\\addlegendentry{{{label}}}"]
        for estimator, dashes, hollow in (("cons", "solid", False), (variant, "densely dotted", True)):
            for setting, (_, _, _, mark) in SETTINGS.items():
                color = f"sc{setting.replace('_', '')}"
                size = ", mark size=2pt" if mark == "triangle*" else ""
                fill = "white" if hollow else color
                edge = color if hollow else "white"
                coordinates = " ".join(f"({x},{y:.4f})" for x, y in points[task, estimator, setting])
                lines.append(f"\\addplot[color={color}, {dashes}, mark={mark}{size}, forget plot, "
                             f"mark options={{solid, draw={edge}, line width=0.35pt, fill={fill}}}] "
                             f"coordinates {{{coordinates}}};")
    lines += ["\\end{groupplot}", "\\end{tikzpicture}", ""]
    return "\n".join(lines)


def main():
    style()
    frame = pd.read_csv(HERE / "deltas.csv")
    sizes = sorted(int(size) for size in frame.n_models.unique())
    for method, suffix in METHODS.items():
        rows = frame[frame.method == method]
        for variant in VARIANTS:
            if variant not in set(rows.estimator):
                continue
            for metric, (name, ylabel) in METRICS.items():
                points = series(rows, variant, metric)
                figure = draw(points, variant, ylabel, sizes, zero=metric.startswith("delta"))
                stem = HERE / f"{variant}{suffix}_{name}"
                figure.savefig(stem.with_suffix(".png"))
                plt.close(figure)
                if metric in TEX_METRICS:
                    stem.with_suffix(".tex").write_text(pgfplots(points, variant, ylabel, sizes))
                print(f"wrote {stem}")


if __name__ == "__main__":
    main()
