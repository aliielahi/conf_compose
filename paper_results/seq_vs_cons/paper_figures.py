"""Paper pgfplots for accuracy and p-Brier against group size, in the style of plot_auarc_agents_estimator.tex."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "paper_content"
SETTINGS = {"voting": ("Before Interaction", "2A78D6", "*"),
            "debate_r1": ("Interaction Round 1", "EB6834", "square*"),
            "debate_r2": ("Interaction Round 2", "1BAF7A", "triangle*")}
# The pooling rule and seq variant the AUARC figure reports.
METHOD, VARIANT = "weighted_sum", "seq_reasoned"
THREE = {"csqa": "CSQA", "truthfulqa": "TruthfulQA", "gsm8k": "GSM8K"}
FIVE = {"csqa": "CSQA", "boolq": "BoolQ", "gsm8k": "GSM8K", "truthfulqa": "TruthfulQA", "gpqa": "GPQA"}
# Filled and hollow triangles read smaller than circles and squares at the same mark size.
TRIANGLE = {False: 2.8, True: 2.2}


def points(frame, tasks, estimators, metric, scale=100):
    return {(task, estimator, setting): list((scale * frame[(frame.task == task) & (frame.estimator == estimator)
                                                          & (frame.setting == setting)]
                                              .groupby("n_models")[metric].mean()).items())
            for task in tasks for estimator in estimators for setting in SETTINGS}


def limits(values, zero):
    low, high = (min(0, *values), max(0, *values)) if zero else (min(values), max(values))
    pad = high - low
    return low - 0.08 * pad, high + 0.08 * pad


def plot_line(points, hollow):
    lines = []
    for setting, (_, _, mark) in SETTINGS.items():
        color = f"sc{setting.replace('_', '')}"
        size = f", mark size={TRIANGLE[hollow]}pt" if mark == "triangle*" else ""
        style, fill, edge = ("densely dotted", "white", color) if hollow else ("solid", color, "white")
        coordinates = " ".join(f"({x},{y:.4f})" for x, y in points[setting])
        lines.append(f"\\addplot[color={color}, {style}, mark={mark}{size}, forget plot, "
                     f"mark options={{solid, draw={edge}, line width=0.35pt, fill={fill}}}] coordinates {{{coordinates}}};")
    return lines


def figure(series, tasks, estimators, ylabel, width, sep, line_legend, zero=True,
           line_labels=("Consistency", "Seq. Prob.")):
    """estimators: (key, hollow) pairs; the legend lists line styles, named by line_labels (solid, dotted), only when
    there are two of them. zero: draw and include the no-change line, for gains over the best single agent.
    With more than three panels the x label is too wide to repeat, so it appears once, under the middle one."""
    sizes = sorted({x for values in series.values() for x, _ in values})
    middle = len(tasks) // 2
    xlabel = "xlabel={Number of Agents in Group}"
    shared = len(tasks) > 3
    lines = ["% Needs in the preamble: \\usepackage{pgfplots} \\usepgfplotslibrary{groupplots} \\pgfplotsset{compat=1.17}"]
    lines += [f"\\definecolor{{sc{s.replace('_', '')}}}{{HTML}}{{{color}}}" for s, (_, color, _) in SETTINGS.items()]
    lines += ["\\begin{tikzpicture}",
              f"\\begin{{groupplot}}[group style={{group size={len(tasks)} by 1, horizontal sep={sep}}},",
              f"  scale only axis, width={width}\\textwidth, height=0.19\\textwidth,",
              f"  xmin={sizes[0] - 0.4}, xmax={sizes[-1] + 0.4}, xtick={{{','.join(map(str, sizes))}}},",
              "  axis line style={black!55, line width=0.4pt}, tick style={black!55, line width=0.4pt}, tick align=inside,",
              "  tick label style={font=\\tiny, black!70}, label style={font=\\scriptsize},",
              "  ymajorgrids, grid style={black!7, line width=0.3pt},",
              "  scaled y ticks=false, yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=3},",
              f"  {'' if shared else xlabel + ', '}title style={{font=\\footnotesize, yshift=-3pt, text height=1.6ex, text depth=0.4ex}},",
              "  every axis plot/.append style={line width=0.9pt, mark size=2.0pt}]"]
    for index, (task, name) in enumerate(tasks.items()):
        low, high = limits([y for e, _ in estimators for s in SETTINGS for _, y in series[task, e, s]], zero)
        options = [f"title={{{name}}}", f"ymin={low:.3f}", f"ymax={high:.3f}"]
        if index == 0:
            options.append(f"ylabel={{{ylabel}}}")
        if index == middle and shared:
            options.append(xlabel)
        if index == middle:
            columns = len(SETTINGS) + (2 if line_legend else 0)
            options += [f"legend columns={columns}",
                        "legend style={at={(0.5,1.3)}, anchor=south, draw=none, fill=none, font=\\scriptsize, "
                        "/tikz/every even column/.append style={column sep=5pt}}"]
        lines.append(f"\\nextgroupplot[{', '.join(options)}]")
        if zero:
            lines.append(f"\\addplot[black!45, dashed, line width=0.4pt, forget plot] coordinates "
                         f"{{({sizes[0] - 0.4},0) ({sizes[-1] + 0.4},0)}};")
        if index == middle:
            if line_legend:
                lines += ["\\addlegendimage{black!80, line width=1.0pt}", f"\\addlegendentry{{{line_labels[0]}}}",
                          "\\addlegendimage{black!80, line width=1.0pt, densely dotted}",
                          f"\\addlegendentry{{{line_labels[1]}}}"]
            for setting, (label, _, mark) in SETTINGS.items():
                lines += [f"\\addlegendimage{{only marks, mark={mark}, color=sc{setting.replace('_', '')}, mark size=2pt}}",
                          f"\\addlegendentry{{{label}}}"]
        for estimator, hollow in estimators:
            lines += plot_line({s: series[task, estimator, s] for s in SETTINGS}, hollow)
    lines += ["\\end{groupplot}", "\\end{tikzpicture}", ""]
    return "\n".join(lines)


def main():
    frame = pd.read_csv(HERE / "deltas.csv")
    frame = frame[frame.method == METHOD]
    # The voted answer does not depend on the confidence estimator, so accuracy is drawn once.
    accuracy = figure(points(frame, FIVE, ["cons"], "delta_accuracy"), FIVE, [("cons", False)],
                      r"$\Delta$ Accuracy", 0.152, "0.7cm", line_legend=False)
    brier = figure(points(frame, THREE, ["cons", VARIANT], "p_brier"), THREE,
                   [("cons", False), (VARIANT, True)], r"p-Brier ($\times 100$)", 0.28, "0.8cm", line_legend=True, zero=False)
    (OUT / "plot_acc_agents.tex").write_text(accuracy)
    (OUT / "plot_brier_agent_estiamtor.tex").write_text(brier)
    print(f"wrote {OUT / 'plot_acc_agents.tex'} and {OUT / 'plot_brier_agent_estiamtor.tex'}")


if __name__ == "__main__":
    main()
