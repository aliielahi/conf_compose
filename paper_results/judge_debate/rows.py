"""Main-table "LLM Elicited" rows for debate rounds 1 and 2: the judge's verbalized confidence in the debate's final
answer, Platt-fitted on the fitting questions and scored on the matched questions against the best-agent reference,
exactly like the voting row."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from conf_compose.utils.calibration import PlattCalibrator
from conf_compose.utils.metrics import auarc, brier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VERDICTS = HERE / "judge_debate_verdicts.csv"
# Debate composition run per round: its audits fix the questions, and its atomic.csv the best-agent references.
RUNS = {1: "run_40bca54388122738", 2: "run_12a013aed30bd201"}
TASKS = ("csqa", "boolq", "gsm8k", "truthfulqa", "gpqa")


def group_scores(verdicts, run, round_index):
    """Per group: judge p-Brier and AUARC on the matched questions, minus the pipeline's best-agent reference."""
    atomic = pd.read_csv(ROOT / f"paper_results/results/debate_protocol/{run}/cons/atomic.csv")
    references = atomic[atomic.method == "both_rounds:logodds_sum"].set_index(["task", "models"])
    rows = []
    for path in sorted((ROOT / "results/debate_composition" / run).glob(f"*/*/round_{round_index}/cons/audit.json")):
        audit = json.loads(path.read_text())
        predictions = {r["id"]: r for r in map(json.loads, path.with_name("predictions.jsonl").open())}
        members = set(audit["models"].split("|"))
        judged = verdicts[(verdicts["round"] == round_index) & (verdicts.task == audit["task"])
                          & (verdicts.panel.map(lambda p: set(p.split("+"))) == members)].set_index("id")
        if judged.empty:
            raise ValueError(f"no judge verdicts for round {round_index} {audit['task']} {audit['models']}")
        mismatched = sum(float(judged.at[q, "correct"]) != float(predictions[q]["correct"]) for q in audit["matched_ids"])
        if mismatched:
            raise ValueError(f"judge labels differ from the debate's on {mismatched} questions: {audit['models']}")
        fit = [q for q in audit["fit_scored_ids"] if pd.notna(judged.at[q, "verbalized"])]
        evaluation = audit["matched_ids"]
        scores = judged.loc[evaluation, "verbalized"].to_numpy(dtype=float)
        if np.isnan(scores).any():
            raise ValueError(f"missing judge confidence on matched questions: {audit['models']}")
        labels = np.asarray([float(predictions[q]["correct"]) for q in evaluation])
        platt = PlattCalibrator().fit(judged.loc[fit, "verbalized"].to_numpy(dtype=float),
                                      [float(predictions[q]["correct"]) for q in fit]).predict(scores)
        reference = references.loc[(audit["task"], audit["models"])]
        rows.append({"round": round_index, "task": audit["task"], "models": audit["models"], "n_eval": len(evaluation),
                     "p_brier": float(brier(platt, labels)), "auarc": float(auarc(np.round(scores, 12), labels)),
                     "delta_p_brier": 100 * (float(brier(platt, labels)) - reference.reference_p_brier),
                     "delta_auarc": 100 * (float(auarc(np.round(scores, 12), labels))
                                           - reference.reference_answer_matched_auarc)})
    return pd.DataFrame(rows)


def cell(values):
    spread = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return f"{values.mean():+.2f} {{\\scriptsize $\\pm$ {spread:.2f}}}"


def main():
    verdicts = pd.read_csv(VERDICTS)
    frame = pd.concat([group_scores(verdicts, run, round_index) for round_index, run in RUNS.items()], ignore_index=True)
    frame.to_csv(HERE / "judge_groups.csv", index=False)
    lines = []
    for round_index in RUNS:
        rows = frame[frame["round"] == round_index]
        counts = rows.groupby("task").size()
        if (counts != 15).any() or set(counts.index) != set(TASKS):
            raise ValueError(f"round {round_index}: expected 15 groups per dataset, got {counts.to_dict()}")
        cells = [cell(rows[rows.task == task][metric]) for task in TASKS for metric in ("delta_p_brier", "delta_auarc")]
        lines.append(f"% Debate round {round_index}\nLLM Elicited & " + " & ".join(cells) + r" \\")
    (HERE / "judge_rows.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
