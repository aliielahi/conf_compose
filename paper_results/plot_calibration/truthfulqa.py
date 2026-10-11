"""One-column TruthfulQA reliability figure, voting above debate, as PDF, PNG and compilable pgfplots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from conf_compose.utils.metrics import ece

HERE = Path(__file__).resolve().parent
TASK = "truthfulqa"
NAME = "calibration_truthfulqa"
BINS = 20
# ACL column width in inches.
WIDTH = 3.03
PROTOCOLS = {"voting": "Voting Accuracy", "debate": "Debate Accuracy"}
METHODS = {
    "single": ("Best Single Agent", "8a8985", "o", "*", 1.8),
    "logodds_sum": ("Log-odds Sum", "e34948", "v", "triangle*", 2.5),
    "logistic_pool": ("Learned Weights", "eda100", "D", "diamond*", 2.2),
    "shared_scale": ("Fitted Scale", "2a78d6", "s", "square*", 1.8),
}
INK, MUTED = "#0b0b0b", "#52514e"


def reliability(confidence, correct):
    """Equal-mass bins, so every point stands for about the same number of questions."""
    edges = np.unique(np.quantile(confidence, np.linspace(0, 1, BINS + 1)))
    index = np.clip(np.searchsorted(edges, confidence, side="right") - 1, 0, len(edges) - 2)
    return np.array([(confidence[index == b].mean(), correct[index == b].mean()) for b in np.unique(index)])


def series(data):
    """Points and ECE per protocol and method, shared by the matplotlib and the pgfplots versions."""
    out = {}
    for protocol in PROTOCOLS:
        for method in METHODS:
            rows = data[(data.protocol == protocol) & (data.task == TASK) & (data.method == method)]
            confidence, correct = rows.confidence.to_numpy(), rows.correct.to_numpy()
            out[protocol, method] = (reliability(confidence, correct), 100 * ece(confidence, correct))
    return out


def limits(points):
    low = min(p[:, :2].min() for p, _ in points.values())
    return max(0.0, np.floor(low * 10) / 10), 1.0


def matplotlib_figure(points):
    plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "legend.fontsize": 5.8,
                         "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "axes.edgecolor": MUTED,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.labelcolor": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
                         "savefig.pad_inches": 0.02, "savefig.dpi": 600})
    low, high = limits(points)
    figure, axes = plt.subplots(2, 1, figsize=(WIDTH, 4.3), sharex=True)
    for axis, (protocol, title) in zip(axes, PROTOCOLS.items()):
        axis.plot([low, high], [low, high], color=MUTED, lw=0.6, ls=(0, (3, 2.5)), zorder=1)
        for method, (label, color, marker, _, _) in METHODS.items():
            values, score = points[protocol, method]
            axis.scatter(values[:, 0], values[:, 1], s=17, marker=marker, color=f"#{color}", edgecolor="white",
                         linewidth=0.35, alpha=0.95, zorder=3, clip_on=False, label=f"{label} ({score:.1f})")
        axis.set(xlim=(low, high), ylim=(low, high), aspect="equal")
        axis.set_ylabel(title)
        axis.legend(loc="lower right", frameon=False, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25)
    axes[1].set_xlabel("Confidence")
    figure.subplots_adjust(hspace=0.08)
    return figure


def coordinates(values):
    return " ".join(f"({x:.4f},{y:.4f})" for x, y in values)


def pgfplots(points):
    """Self-contained: needs \\usepackage{pgfplots} and \\usepgfplotslibrary{groupplots} in the preamble."""
    low, high = limits(points)
    lines = ["% Needs in the preamble: \\usepackage{pgfplots} \\usepgfplotslibrary{groupplots} \\pgfplotsset{compat=1.17}"]
    lines += [f"\\definecolor{{cal{method}}}{{HTML}}{{{color.upper()}}}" for method, (_, color, _, _, _) in METHODS.items()]
    lines += ["\\begin{tikzpicture}",
              "\\begin{groupplot}[group style={group size=1 by 2, vertical sep=12pt, x descriptions at=edge bottom},",
              f"  width=0.9\\columnwidth, height=0.9\\columnwidth, xmin={low}, xmax={high}, ymin={low}, ymax={high},",
              "  axis lines=left, axis line style={-, black!60}, tick style={black!60}, tick label style={font=\\tiny}, label style={font=\\scriptsize},",
              "  legend style={font=\\tiny, draw=none, fill=none, at={(0.98,0.02)}, anchor=south east, row sep=-2pt},",
              "  legend cell align=left, xlabel={Confidence}]"]
    for protocol, title in PROTOCOLS.items():
        lines.append(f"\\nextgroupplot[ylabel={{{title}}}]")
        lines.append(f"\\addplot[black!60, dashed, thin, forget plot] coordinates {{({low},{low}) ({high},{high})}};")
        for method, (label, _, _, mark, size) in METHODS.items():
            values, score = points[protocol, method]
            lines.append(f"\\addplot[only marks, mark={mark}, color=cal{method}, mark size={size}pt, mark options={{draw=white, line width=0.2pt}}] "
                         f"coordinates {{{coordinates(values)}}};")
            lines.append(f"\\addlegendentry{{{label} ({score:.1f})}}")
    lines += ["\\end{groupplot}", "\\end{tikzpicture}", ""]
    return "\n".join(lines)


def main():
    points = series(pd.read_csv(HERE / "confidences.csv.gz"))
    figure = matplotlib_figure(points)
    for suffix in ("pdf", "png"):
        figure.savefig(HERE / f"{NAME}.{suffix}")
    (HERE / f"{NAME}.tex").write_text(pgfplots(points))
    print(f"wrote {HERE / NAME}.pdf/.png/.tex")


if __name__ == "__main__":
    main()
