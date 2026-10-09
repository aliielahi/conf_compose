import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from conf_compose.utils import metrics
from conf_compose.utils.calibration import PlattCalibrator, fit_temperature, temperature_scale
from significance import dataset_sign_tests
from tables import FULL_METRICS, write_csv, write_significance_tables, write_tables

RAW_METRICS = ("accuracy", "ece", "auarc", "auroc", "brier", "nll")

DEFAULT_RUN = ROOT / "baselines/results/voting_adapter/a1fc8f63af8e71c5"
LABELS = {"cagecal_iid": "CAGE-CAL (IID)", "cagecal_iid_betasb": "CAGE-CAL (IID + BetaSB)"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_digest(ids):
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def cell_key(row):
    return row["task"], row["models"], row["estimator"]


def append_results(rows, cells, directory=DEFAULT_RUN, score="both"):
    if score not in ("raw", "betasb", "both"):
        raise ValueError(f"unknown CAGE-CAL score: {score}")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    calibration = json.loads((directory / "calibration.json").read_text())
    saved_cells = {cell_key(cell): cell for cell in json.loads((directory / "paper_tables/audit.json").read_text())}
    with (directory / "paper_tables/cagecal_metrics.csv").open() as handle:
        saved_metrics = {(cell_key(row), row["method"]): row for row in csv.DictReader(handle)}

    def predictions_at(path):
        result = {}
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                prediction = json.loads(line)
                key = prediction["task"], prediction["models"], prediction["id"]
                if key in result:
                    raise ValueError(f"duplicate CAGE-CAL prediction: {key}")
                result[key] = prediction
        return result

    predictions = predictions_at(directory / "predictions.jsonl")
    validation = predictions_at(directory / "validation_predictions.jsonl")
    original = [row for row in rows if not row["method"].startswith("cagecal_")]
    references = {cell_key(row): row for row in original if row["method"] == "best_solo"}
    methods = [("cagecal_iid", "raw"), ("cagecal_iid_betasb", "betasb")]
    if score != "both":
        methods = [methods[0] if score == "raw" else methods[1]]
    appended = []
    for cell in cells:
        key = cell_key(cell)
        saved = saved_cells[key]
        for field in ("matched_ids", "fit_ids_hash", "eval_ids_hash", "tie_break", "tie_seed"):
            if cell[field] != saved[field]:
                raise ValueError(f"CAGE-CAL {field} mismatch: {key}")
        split = manifest["splits"][cell["task"]]
        train, val_ids, evaluation = (set(split[name]) for name in ("train", "validation", "evaluation"))
        if train & val_ids or (train | val_ids) & evaluation:
            raise ValueError(f"CAGE-CAL split overlap: {key}")
        if ids_digest(train | val_ids) != cell["fit_ids_hash"] or ids_digest(evaluation) != cell["eval_ids_hash"]:
            raise ValueError(f"CAGE-CAL fitting/evaluation split mismatch: {key}")
        selected = [predictions[(cell["task"], cell["models"], question)] for question in cell["matched_ids"]]
        for question, prediction in zip(cell["matched_ids"], selected):
            if prediction["target"] != cell["selected_targets"][question] or prediction["correct"] not in (0, 1):
                raise ValueError(f"CAGE-CAL target mismatch or invalid label: {key}/{question}")
        val = [prediction for (task, models, question), prediction in validation.items()
               if (task, models) == key[:2]]
        if not val or not set(prediction["id"] for prediction in val) <= val_ids:
            raise ValueError(f"CAGE-CAL validation IDs mismatch: {key}")
        for method, field in methods:
            probabilities = np.round(np.asarray([item[field] for item in selected], dtype=float), 12)
            if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError(f"invalid CAGE-CAL confidence: {key}")
            correct = np.asarray([item["correct"] for item in selected])
            values = {metric: float(getattr(metrics, metric)(probabilities, correct))
                      for metric in RAW_METRICS if metric != "accuracy"}
            values["accuracy"] = float(correct.mean())
            if not math.isclose(values["accuracy"], cell["vote_accuracy"], abs_tol=1e-12):
                raise ValueError(f"CAGE-CAL voting accuracy mismatch: {key}")
            exported = saved_metrics[key, method]
            for metric, value in values.items():
                expected = float(exported[metric]) if exported[metric] else float("nan")
                if not (math.isnan(value) and math.isnan(expected)) and not math.isclose(value, expected, abs_tol=1e-12):
                    raise ValueError(f"CAGE-CAL saved {metric} does not match predictions: {key}")
            temperature = None
            t_values = {"t_brier": None, "t_ece": None, "p_brier": None, "p_ece": None}
            if field == "raw":
                temperature = (fit_temperature([item["raw"] for item in val], [item["correct"] for item in val])
                               if len({item["correct"] for item in val}) > 1 else 1.0)
                if not math.isclose(temperature, float(exported["output_temperature"]), abs_tol=1e-5):
                    raise ValueError(f"CAGE-CAL output temperature mismatch: {key}")
                temperature = float(exported["output_temperature"])
                calibrated = temperature_scale(probabilities, temperature)
                t_values = {"t_brier": float(metrics.brier(calibrated, correct)),
                            "t_ece": float(metrics.ece(calibrated, correct))}
                for metric, value in t_values.items():
                    if not math.isclose(value, float(exported[metric]), abs_tol=1e-12):
                        raise ValueError(f"CAGE-CAL saved {metric} does not match validation fit: {key}")
                if len({item["correct"] for item in val}) > 1:
                    platt = PlattCalibrator().fit([item["raw"] for item in val],
                                                  [item["correct"] for item in val]).predict(probabilities)
                    t_values.update(p_brier=float(metrics.brier(platt, correct)), p_ece=float(metrics.ece(platt, correct)))
            row = dict(references[key], method=method, judge_approximate=False, judge_target_mismatches=0)
            row["output_temperature"] = temperature
            values.update(t_values)
            values["answer_matched_auarc"] = values["auarc"]
            for metric, value in values.items():
                value = value if value is not None and math.isfinite(value) else None
                reference = row.get(f"reference_{metric}")
                row[metric] = value
                row[f"delta_{metric}"] = value - reference if value is not None and reference is not None else None
            appended.append(row)
    files = ("manifest.json", "predictions.jsonl", "validation_predictions.jsonl", "calibration.json",
             "paper_tables/audit.json", "paper_tables/cagecal_metrics.csv")
    provenance = {"directory": str(directory.resolve()), "run_identity": manifest["identity"], "score": score,
                  "methods": [name for name, _ in methods], "seeds": manifest["config"]["seeds"],
                  "n_cells": len(appended), "calibration": calibration,
                  "sources": {name: digest(directory / name) for name in files},
                  "temperature_metrics": "raw CAGE score only; output temperature fitted on disjoint validation predictions",
                  "same_prediction_across_estimators": "evaluated on each estimator's existing matched question mask"}
    return original + appended, {method: LABELS[method] for method, _ in methods}, provenance


def read_rows(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if "judge_approximate" in row:
            row["judge_approximate"] = row["judge_approximate"].lower() in ("true", "1")
        for key in ("n_models", "n_matched", "judge_target_mismatches"):
            if key in row:
                row[key] = int(row[key])
        for key in row:
            if key == "coverage" or key in FULL_METRICS or key.startswith("delta_") or key.startswith("reference_") and key != "reference_mode" and not key.endswith("_model"):
                row[key] = float(row[key]) if row[key] else None
    return rows


def main():
    parser = argparse.ArgumentParser(description="Refresh saved voting tables with audited CAGE-CAL predictions; no inference or refitting")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "paper_results/results/voting_protocol")
    parser.add_argument("--cagecal-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--score", choices=("raw", "betasb", "both"), default="both")
    args = parser.parse_args()
    directory = args.report_dir
    manifest = json.loads((directory / "manifest.json").read_text())
    cells = json.loads((directory / "audit.json").read_text())
    rows, added_labels, provenance = append_results(read_rows(directory / "atomic.csv"), cells, args.cagecal_dir, args.score)
    labels = manifest.get("method_labels")
    if labels is None:
        table = (directory / f"{manifest['arguments']['estimators'][0]}/all_delta.txt").read_text().splitlines()
        names = [line.split(" | ")[0].strip() for line in table if " | " in line][1:]
        methods = list(dict.fromkeys(row["method"] for row in rows if not row["method"].startswith("cagecal_")))
        names = [name for name in names if not name.startswith("CAGE-CAL")]
        if len(names) != len(methods):
            raise ValueError("existing method labels do not match atomic rows")
        labels = dict(zip(methods, names))
    labels = {key: label for key, label in labels.items() if not key.startswith("cagecal_")}
    labels.update(added_labels)
    estimators = manifest["arguments"]["estimators"]
    tasks = manifest["arguments"]["tasks"]
    files = write_tables(directory, rows, cells, estimators, labels, tasks)
    if manifest["arguments"].get("significance"):
        tested = {key: value for key, value in labels.items() if not (manifest.get("judge_policy") == "approximate" and key.startswith("judge:"))}
        tests = dataset_sign_tests(rows, cells, estimators, tested, tasks)
        write_csv(directory / "significance.csv", tests)
        files.extend(write_significance_tables(directory, tests, estimators, tested))
    write_csv(directory / "atomic.csv", rows)
    manifest.update(cagecal=provenance, tables=files, method_labels=labels)
    manifest["report_code"] = {str(path): digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    manifest.pop("table_render_source", None)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Added {provenance['n_cells']} CAGE-CAL rows; wrote {len(files)} tables to {directory}")


if __name__ == "__main__":
    main()
