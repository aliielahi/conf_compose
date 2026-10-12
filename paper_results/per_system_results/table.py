"""Appendix tables of absolute per-group results, one per confidence estimator, shading each learned-weight cell
by its gain over the best agent."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PANELS = json.loads((ROOT / "src/conf_compose/constants.json").read_text())["composition_panels"]
TASKS = {"csqa": "CSQA", "boolq": "BoolQ", "gsm8k": "GSM8K", "truthfulqa": "TruthfulQA", "gpqa": "GPQA"}
SETTINGS = {"voting": "Before Interaction", "debate_r1": "Interaction Round 1", "debate_r2": "Interaction Round 2"}
ESTIMATORS = {"cons": "consistency", "seq_reasoned": "sequence probability (after reasoning)"}
# Column order inside each metric block: the best agent, then learned weights in probability and log-odds space.
RULES = {"weighted_average": "Prob.", "weighted_sum": "Log-odds"}
# Shading saturates at this many points of gain (or loss) over the best agent.
SATURATION = {"auarc": 10.0, "p_brier": 4.0}
STRONGEST = 55


def group_number(models):
    members = set(models.split("|"))
    return next(number for number, panel in enumerate(PANELS, 1) if set(panel) == members)


def shade(gain, metric):
    """Green for a gain over the best agent, orange for a loss, deeper the larger it is."""
    strength = int(round(STRONGEST * min(abs(gain) / SATURATION[metric], 1.0)))
    if strength < 3:
        return ""
    return f"\\cellcolor{{{'gain' if gain > 0 else 'loss'}!{strength}}}"


def cells(rows, setting):
    """Accuracy, then AUARC and p-Brier for best agent / probability / log-odds, all x100."""
    best = rows[(setting, "weighted_sum")]
    out = [f"{100 * best.accuracy:.2f}"]
    for metric, reference, sign in (("auarc", "reference_auarc", 1), ("p_brier", "reference_p_brier", -1)):
        out.append(f"{100 * best[reference]:.2f}")
        for rule in RULES:
            value = 100 * rows[(setting, rule)][metric]
            out.append(f"{shade(sign * (value - 100 * best[reference]), metric)}{value:.2f}")
    return out


def table(frame, estimator):
    frame = frame[frame.estimator == estimator].assign(group=lambda f: f.models.map(group_number))
    blocks = len(SETTINGS)
    spec = "cc" + "|c|ccc|ccc" * blocks
    lines = ["% Needs: \\usepackage{booktabs,multirow,graphicx} \\usepackage[table]{xcolor}",
             "\\definecolor{gain}{HTML}{1BAF7A}", "\\definecolor{loss}{HTML}{EB6834}",
             "\\begin{table*}[p]", "\\centering", "\\tiny", "\\setlength{\\tabcolsep}{2.2pt}",
             "\\renewcommand{\\arraystretch}{1.15}", "\\resizebox{\\textwidth}{!}{%",
             f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             "\\multirow{3}{*}{\\textbf{Dataset}} & \\multirow{3}{*}{\\textbf{Group}} & "
             + " & ".join(f"\\multicolumn{{7}}{{c{'|' if i < blocks - 1 else ''}}}{{\\textbf{{{name}}}}}"
                          for i, name in enumerate(SETTINGS.values())) + " \\\\",
             " & & " + " & ".join(f"\\multirow{{2}}{{*}}{{Acc}} & \\multicolumn{{3}}{{c|}}{{AUARC $\\uparrow$}} & "
                                  f"\\multicolumn{{3}}{{c{'|' if i < blocks - 1 else ''}}}{{p-Brier $\\downarrow$}}"
                                  for i in range(blocks)) + " \\\\",
             " & & " + " & ".join(" & ".join(["", "Best", *RULES.values(), "Best", *RULES.values()])
                                  for _ in range(blocks)) + " \\\\",
             "\\midrule"]
    indexed = frame.set_index(["task", "group", "setting", "method"]).sort_index()
    for index, (task, name) in enumerate(TASKS.items()):
        numbers = sorted(frame[frame.task == task].group.unique())
        for position, number in enumerate(numbers):
            picked = {(setting, rule): indexed.loc[(task, number, setting, rule)] for setting in SETTINGS for rule in RULES}
            label = (f"\\multirow{{{len(numbers)}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\textbf{{{name}}}}}}}"
                     if position == 0 else "")
            values = [value for setting in SETTINGS for value in cells(picked, setting)]
            lines.append(f"{label} & {number} & " + " & ".join(values) + " \\\\")
        lines.append("\\midrule" if index < len(TASKS) - 1 else "\\bottomrule")
    lines += ["\\end{tabular}}",
              f"\\caption{{\\textbf{{Per-group results with {ESTIMATORS[estimator]} confidence.}} Absolute matched "
              "accuracy, accuracy-matched AUARC and Platt-calibrated Brier ($\\times 100$) for each of the 15 model groups "
              "(Table~\\ref{tab:model_groups}) on every dataset. \\emph{Best} is the single agent with the highest "
              "accuracy on the fitting split, scored on the group's selected answer; \\emph{Prob.} and \\emph{Log-odds} "
              "are the learned weights in probability and log-odds space. Interaction rounds pool each agent's round-0 "
              "and post-debate confidence. Shading marks the gain (green) or loss (orange) of a learned-weight rule "
              f"over \\emph{{Best}}, saturating at {SATURATION['auarc']:g} AUARC and {SATURATION['p_brier']:g} p-Brier "
              "points.}",
              f"\\label{{tab:per_group_{estimator}}}", "\\end{table*}", ""]
    return "\n".join(lines)


def main():
    frame = pd.read_csv(HERE / "systems.csv")
    for estimator in ESTIMATORS:
        path = HERE / f"per_group_{estimator}.tex"
        path.write_text(table(frame, estimator))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
