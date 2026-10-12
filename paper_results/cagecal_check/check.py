"""CAGE-CAL fairness check: reproduce its table cells, and see how its p-Brier depends on the calibration data it gets."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from conf_compose.utils import metrics
from conf_compose.utils.calibration import PlattCalibrator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = {"voting": (ROOT / "baselines/results/voting_adapter/a1fc8f63af8e71c5",
                   ROOT / "paper_results/results/voting_protocol", "cagecal_iid"),
        "debate_r1": (ROOT / "baselines/results/debate_adapter/71259088b2f0e984",
                      ROOT / "paper_results/results/debate_protocol/run_40bca54388122738/cons", "cagecal_debate")}
TASKS = ("csqa", "boolq", "gsm8k", "truthfulqa", "gpqa")


def read(path):
    rows = [json.loads(line) for line in path.open() if line.strip()]
    return {(r["task"], r["models"], r["id"]): r for r in rows}


def platt(fit_scores, fit_labels, scores, labels):
    if len(set(fit_labels)) < 2:
        return None
    return float(metrics.brier(PlattCalibrator().fit(fit_scores, fit_labels).predict(scores), labels))


def cells(setting):
    """Per group: CAGE-CAL p-Brier with per-group validation Platt (the table), with Platt on every group's
    validation for the task (the model is shared across groups), and the uncalibrated Brier."""
    adapter, report, method = RUNS[setting]
    predictions, validation = read(adapter / "predictions.jsonl"), read(adapter / "validation_predictions.jsonl")
    atomic = pd.read_csv(report / "atomic.csv")
    if "estimator" in atomic:
        atomic = atomic[atomic.estimator == "cons"]
    audits = {(a["task"], a["models"]): a for a in json.loads((ROOT / ("paper_results/results/voting_protocol/audit.json"
              if setting == "voting" else f"{report}/audit.json")).read_text())
              if a.get("estimator", "cons") == "cons"}
    rows = []
    for (task, models), audit in audits.items():
        selected = [predictions[task, models, q] for q in audit["matched_ids"]]
        scores, labels = [p["raw"] for p in selected], [float(p["correct"]) for p in selected]
        own = [v for (t, m, _), v in validation.items() if (t, m) == (task, models)]
        pooled = [v for (t, _, _), v in validation.items() if t == task]
        table = atomic[(atomic.task == task) & (atomic.models == models) & (atomic.method == method)].iloc[0]
        rows.append({"setting": setting, "task": task, "models": models, "n_validation": len(own),
                     "table_p_brier": table.p_brier, "reference_p_brier": table.reference_p_brier,
                     "group_platt": platt([v["raw"] for v in own], [v["correct"] for v in own], scores, labels),
                     "task_platt": platt([v["raw"] for v in pooled], [v["correct"] for v in pooled], scores, labels),
                     "uncalibrated": float(metrics.brier(np.asarray(scores), np.asarray(labels))),
                     "table_auarc": table.auarc, "auarc": float(metrics.auarc(np.round(np.asarray(scores), 12),
                                                                               np.asarray(labels)))})
    return pd.DataFrame(rows)


def main():
    frame = pd.concat([cells(setting) for setting in RUNS], ignore_index=True)
    frame.to_csv(HERE / "cagecal_cells.csv", index=False)
    print("reproduction: max |p-Brier - table|", float((frame.group_platt - frame.table_p_brier).abs().max()),
          " max |AUARC - table|", float((frame.auarc - frame.table_auarc).abs().max()))
    summary = frame.assign(**{f"d_{c}": 100 * (frame[c] - frame.reference_p_brier)
                              for c in ("group_platt", "task_platt", "uncalibrated")})
    print(summary.groupby(["setting", "task"])[["n_validation", "d_group_platt", "d_task_platt", "d_uncalibrated"]]
          .mean().round(2).reindex(TASKS, level="task").to_string())


if __name__ == "__main__":
    main()
