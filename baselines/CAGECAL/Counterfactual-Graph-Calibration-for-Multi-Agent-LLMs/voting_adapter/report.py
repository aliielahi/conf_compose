"""Add CAGE rows to copies of the paper tables, using their exact masks."""
from __future__ import annotations

import csv
import json

import numpy as np

from .data import load_module, metric_module, safe_output

METRICS = ("accuracy", "ece", "auarc", "auroc", "brier", "nll")
TEMPERATURE_METRICS = ("t_brier", "t_ece")

LABELS = {"best_solo": "Best solo", "mean": "Arithmetic mean", "logodds_sum": "Log-odds sum",
          "logodds_mean": "Log-odds mean", "shared_rho": "Shared rho", "shared_scale": "Shared scale",
          "kahn": "Kahn", "blp": "BLP", "logistic_pool": "Logistic pool", "blp_equal": "BLP equal",
          "kahn_diagonal": "Kahn diagonal", "cagecal_iid": "CAGE-Cal (IID adaptation)",
          "cagecal_iid_betasb": "CAGE-Cal (IID adaptation + BetaSB)",
          "cagecal_debate": "CAGE-Cal (paired debate)",
          "cagecal_debate_betasb": "CAGE-Cal (paired debate + BetaSB)"}


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
                    if (r["task"], r["models"], r["estimator"]) in keys and not r["method"].startswith("cagecal_")]
    for r in original:
        for key in ("n_models", "n_matched"):
            r[key] = int(r[key])
        numeric = ["coverage"] + [key for metric in METRICS + TEMPERATURE_METRICS
                   for key in (metric, "reference_" + metric, "delta_" + metric, "reference_" + metric + "_accuracy")]
        for key in numeric:
            if key in r:
                r[key] = float(r[key]) if r[key] != "" else None
    calibration = load_module(bundle["project"] / "src/conf_compose/utils/calibration.py", "cage_output_temperature")
    validation = {}
    val_path = prediction_path.with_name("validation_predictions.jsonl")
    if val_path.exists():
        expected_val = {(r["task"], r["models"], r["id"]): r for r in bundle["rows"] if r["split"] == "validation"}
        seen = set()
        with val_path.open() as handle:
            for line in handle:
                r = json.loads(line)
                key = (r["task"], r["models"], r["id"])
                if key in seen or key not in expected_val or key in expected:
                    raise ValueError("Invalid validation IDs or validation/evaluation leakage")
                if any(r[k] != expected_val[key][k] for k in ("target", "correct")):
                    raise ValueError("Validation target/label differs from audit")
                if not np.isfinite(r["raw"]) or not 0 <= r["raw"] <= 1:
                    raise ValueError("Invalid validation probability")
                seen.add(key)
                validation.setdefault(key[:2], []).append(r)
        if seen != set(expected_val):
            raise ValueError("Incomplete validation predictions")
    temperatures = []
    appended, metrics_rows = [], []
    for cell in cells:
        task, models, estimator = cell["task"], cell["models"], cell["estimator"]
        sample = next(r for r in original if (r["task"], r["models"], r["estimator"]) == (task, models, estimator))
        rows = [predictions[(task, models, q)] for q in cell["matched_ids"]]
        y = np.asarray([r["correct"] for r in rows])
        prefix = "cagecal_debate" if bundle.get("protocol") == "debate" else "cagecal_iid"
        for name, field in ((prefix, "raw"), (prefix + "_betasb", "betasb")):
            p = np.round(np.asarray([r[field] for r in rows]), 12)
            if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise ValueError("Invalid predicted confidence")
            score = {key: getattr(methods, key)(p, y) for key in METRICS if key != "accuracy"}
            accuracy = float(y.mean())
            if abs(accuracy - cell["vote_accuracy"]) > 1e-12:
                raise ValueError("CAGE changed vote accuracy")
            score["accuracy"] = cell.get("accuracy_all", accuracy)
            score = {key: float(value) if np.isfinite(value) else None for key, value in score.items()}
            temperature = None
            t_scores = dict(t_brier=None, t_ece=None)
            val = validation.get((task, models), [])
            if field == "raw" and val:
                if len({r["correct"] for r in val}) < 2:
                    temperature, status = 1.0, "identity: one validation class"
                else:
                    temperature = calibration.fit_temperature([r["raw"] for r in val], [r["correct"] for r in val])
                    status = "fitted_on_internal_validation"
                tp = calibration.temperature_scale(p, temperature)
                t_scores = dict(t_brier=float(methods.brier(tp, y)), t_ece=float(methods.ece(tp, y)))
                temperatures.append(dict(task=task, models=models, estimator=estimator, temperature=temperature,
                                         n_validation=len(val), status=status))
            row = dict(sample)
            if "output_temperature" in row:
                row["output_temperature"] = temperature
            for metric in TEMPERATURE_METRICS:
                # Never inherit another method's temperature-calibrated values.
                if metric in row:
                    row[metric] = None
                if "delta_" + metric in row:
                    row["delta_" + metric] = None
            row["method"] = name
            if "status" in row:
                row["status"] = "trained"
            for metric, value in {**score, **t_scores}.items():
                if metric in row:
                    row[metric] = value
                delta = "delta_" + metric
                if delta in row:
                    reference = sample.get("reference_" + metric)
                    row[delta] = value - reference if value is not None and reference is not None else None
            if "judge_approximate" in row:
                row["judge_approximate"] = False
                row["judge_target_mismatches"] = 0
            appended.append(row)
            metrics_rows.append(dict(task=task, models=models, estimator=estimator, method=name,
                n=len(y), coverage=cell["coverage"], **score, **t_scores, output_temperature=temperature))
    tables = load_module(bundle["project"] / "paper_results/codes/voting_protocol/tables.py", "voting_original_tables")
    tables.write_csv(output / "atomic.csv", original + appended)
    tables.write_csv(output / "cagecal_metrics.csv", metrics_rows)
    (output / "audit.json").write_text(json.dumps(cells, indent=2))
    labels = {r["method"]: LABELS.get(r["method"], r["method"]) for r in original + appended}
    for row in original:
        if row["method"].startswith("judge:") and str(row.get("judge_approximate", "")).lower() in ("true", "1"):
            labels[row["method"]] = LABELS.get(row["method"], row["method"]) + " (approx.)"
    tables.write_tables(output, original + appended, cells, sorted({c["estimator"] for c in cells}),
                        labels, sorted({c["task"] for c in cells}))
    (output / "temperatures.json").write_text(json.dumps(temperatures, indent=2))
    # The shared formatter uses voting wording; adapt only our generated copies.
    if bundle.get("protocol") == "debate":
        for path in [*output.rglob("*.txt"), *output.rglob("*.tex")]:
            text = path.read_text().replace("Voting protocol", "Debate protocol").replace("IID voting adaptation", "paired debate adaptation").replace("IID adaptation;", "paired debate adaptation;")
            path.write_text(text)
    return metrics_rows
