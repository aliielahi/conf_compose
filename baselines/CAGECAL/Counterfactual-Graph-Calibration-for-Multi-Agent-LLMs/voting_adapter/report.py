"""Add CAGE rows to copies of the paper tables, using their exact masks."""
from __future__ import annotations

import csv
import json

import numpy as np

from .data import load_module, metric_module, safe_output

LABELS = {"best_solo": "Best solo", "mean": "Arithmetic mean", "logodds_sum": "Log-odds sum",
          "logodds_mean": "Log-odds mean", "shared_rho": "Shared rho", "shared_scale": "Shared scale",
          "kahn": "Kahn", "blp": "BLP", "logistic_pool": "Logistic pool", "blp_equal": "BLP equal",
          "kahn_diagonal": "Kahn diagonal", "cagecal_iid": "CAGE-Cal (IID adaptation)",
          "cagecal_iid_betasb": "CAGE-Cal (IID adaptation + BetaSB)"}


def make_reports(bundle, prediction_path, output):
    output = safe_output(output)
    output.mkdir(parents=True, exist_ok=True)
    predictions = {}
    with prediction_path.open() as handle:
        for line in handle:
            r = json.loads(line)
            key = (r["task"], r["models"], r["id"])
            if key in predictions:
                raise ValueError(f"Duplicate prediction {key}")
            predictions[key] = r
    expected = {(r["task"], r["models"], r["id"]): r for r in bundle["rows"] if r["split"] == "evaluation"}
    if set(predictions) != set(expected):
        raise ValueError("Prediction IDs differ from the audited evaluation set")
    for key, p in predictions.items():
        if p["target"] != expected[key]["target"] or p["correct"] != expected[key]["correct"]:
            raise ValueError(f"Prediction changed fixed vote target/label: {key}")
    methods = metric_module(bundle["project"])
    cells = bundle["cells"]
    keys = {(c["task"], c["models"], c["estimator"]) for c in cells}
    with bundle["atomic_path"].open() as handle:
        original = [r for r in csv.DictReader(handle)
                    if (r["task"], r["models"], r["estimator"]) in keys]
    for r in original:
        for key in ("n_models", "n_matched"):
            r[key] = int(r[key])
        for key in ("coverage", "accuracy", "ece", "reference_ece", "delta_ece", "reference_ece_accuracy",
                    "auarc", "reference_auarc", "delta_auarc", "reference_auarc_accuracy"):
            if r[key] != "":
                r[key] = float(r[key])
    appended, metrics_rows = [], []
    for cell in cells:
        task, models, estimator = cell["task"], cell["models"], cell["estimator"]
        sample = next(r for r in original if (r["task"], r["models"], r["estimator"]) == (task, models, estimator))
        rows = [predictions[(task, models, q)] for q in cell["matched_ids"]]
        y = np.asarray([r["correct"] for r in rows])
        for name, field in (("cagecal_iid", "raw"), ("cagecal_iid_betasb", "betasb")):
            p = np.round(np.asarray([r[field] for r in rows]), 12)
            if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise ValueError("Invalid predicted confidence")
            score = {key: getattr(methods, key)(p, y) for key in ("ece", "auarc", "auroc", "brier", "nll")}
            accuracy = float(y.mean())
            if abs(accuracy - cell["vote_accuracy"]) > 1e-12:
                raise ValueError("CAGE changed vote accuracy")
            row = dict(sample)
            row.update(method=name, accuracy=accuracy, ece=score["ece"], auarc=score["auarc"],
                       delta_ece=score["ece"] - sample["reference_ece"],
                       delta_auarc=score["auarc"] - sample["reference_auarc"])
            appended.append(row)
            metrics_rows.append(dict(task=task, models=models, estimator=estimator, method=name,
                n=len(y), coverage=cell["coverage"], accuracy=accuracy, **score))
    tables = load_module(bundle["project"] / "paper_results/codes/voting_protocol/tables.py", "voting_original_tables")
    tables.write_csv(output / "atomic.csv", original + appended)
    tables.write_csv(output / "cagecal_metrics.csv", metrics_rows)
    (output / "audit.json").write_text(json.dumps(cells, indent=2))
    labels = {r["method"]: LABELS.get(r["method"], r["method"]) for r in original + appended}
    tables.write_tables(output, original + appended, cells, sorted({c["estimator"] for c in cells}),
                        labels, sorted({c["task"] for c in cells}))
    return metrics_rows
