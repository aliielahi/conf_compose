"""Paper-table rows for the proposed method set: Δ against the best single agent, mean ± 95% CI over groups."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TASKS = ("csqa", "boolq", "gsm8k", "truthfulqa", "gpqa")
SETTINGS = {"voting": "Voting", "debate_r1": "Debate, round 1", "debate_r1_both": "Debate, round 1, both rounds",
            "debate_r2": "Debate, round 2", "debate_r2_both": "Debate, round 2, both rounds"}
ROWS = {"mean": "Arithmetic Average", "blp": "Learned Weights (prob), current",
        "weighted_average_1se": "Learned Weights (prob), proposed", "logodds_sum": "Log-odds Sum",
        "shared_scale": "Fitted Scale", "logistic_pool": "Learned Weights (logit), current",
        "weighted_sum_1se": "Learned Weights (logit), proposed"}
DEBATE_RUNS = {"debate_r1": "run_40bca54388122738", "debate_r2": "run_12a013aed30bd201"}


def references():
    """Best-single-agent p-Brier and same-answer AUARC per group, recovered from the pipeline's deltas."""
    voting = pd.read_csv(ROOT / "paper_results/results/voting_protocol/atomic.csv")
    frames = [voting[(voting.estimator == "cons") & (voting.method == "logodds_sum")].assign(setting="voting")]
    for setting, run in DEBATE_RUNS.items():
        debate = pd.read_csv(ROOT / f"paper_results/results/debate_protocol/{run}/cons/atomic.csv")
        for suffix, method in (("", "logodds_sum"), ("_both", "both_rounds:logodds_sum")):
            frames.append(debate[debate.method == method].assign(setting=setting + suffix))
    frame = pd.concat(frames)
    return pd.DataFrame({"setting": frame.setting, "task": frame.task, "models": frame.models,
                         "reference_p_brier": frame.p_brier - frame.delta_p_brier,
                         "reference_auarc": frame.answer_matched_auarc - frame.delta_answer_matched_auarc})


def cell(values):
    spread = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return f"{values.mean():+.2f} {{\\scriptsize $\\pm$ {spread:.2f}}}"


def main():
    scores = pd.read_csv(HERE / "scores.csv").merge(references(), on=["setting", "task", "models"], validate="many_to_one")
    scores["d_brier"] = 100 * (scores.p_brier - scores.reference_p_brier)
    scores["d_auarc"] = 100 * (scores.auarc - scores.reference_auarc)
    lines = []
    for setting, title in SETTINGS.items():
        lines += ["", f"% {title}"]
        for method, label in ROWS.items():
            rows = scores[(scores.setting == setting) & (scores.method == method)]
            cells = [cell(rows[rows.task == task][metric]) for task in TASKS for metric in ("d_brier", "d_auarc")]
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    (HERE / "table_rows.tex").write_text("\n".join(lines).strip() + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
