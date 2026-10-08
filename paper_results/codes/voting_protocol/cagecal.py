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
from significance import dataset_sign_tests
from tables import FULL_METRICS, write_csv, write_significance_tables, write_tables

DEFAULT_RUN = ROOT / "baselines/CAGECAL/Counterfactual-Graph-Calibration-for-Multi-Agent-LLMs/results/voting_adapter/5f0d2df353f60fe6"
LABELS = {"cagecal_iid": "CAGE-CAL (IID)", "cagecal_iid_betasb": "CAGE-CAL (IID + BetaSB)"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_digest(ids):
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def cell_key(row):
    return row["task"], row["models"], row["estimator"]


def append_results(rows, cells, directory=DEFAULT_RUN, score="betasb"):
    if score not in ("raw", "betasb"):
        raise ValueError(f"unknown CAGE-CAL score: {score}")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    calibration = json.loads((directory / "calibration.json").read_text())
    saved_cells = {cell_key(cell): cell for cell in json.loads((directory / "paper_tables/audit.json").read_text())}
    with (directory / "paper_tables/cagecal_metrics.csv").open() as handle:
        saved_metrics = {(cell_key(row), row["method"]): row for row in csv.DictReader(handle)}
    predictions = {}
    with (directory / "predictions.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            prediction = json.loads(line)
            key = prediction["task"], prediction["models"], prediction["id"]
            if key in predictions:
                raise ValueError(f"duplicate CAGE-CAL prediction: {key}")
            predictions[key] = prediction
    original = [row for row in rows if not row["method"].startswith("cagecal_")]
    references = {cell_key(row): row for row in original if row["method"] == "best_solo"}
    method = "cagecal_iid_betasb" if score == "betasb" else "cagecal_iid"
    appended = []
    for cell in cells:
        key = cell_key(cell)
        saved = saved_cells[key]
        for field in ("matched_ids", "fit_ids_hash", "eval_ids_hash", "tie_break", "tie_seed"):
            if cell[field] != saved[field]:
                raise ValueError(f"CAGE-CAL {field} mismatch: {key}; regenerate baseline for this evaluation")
        split = manifest["splits"][cell["task"]]
        train, validation, evaluation = (set(split[name]) for name in ("train", "validation", "evaluation"))
        if train & validation or (train | validation) & evaluation:
            raise ValueError(f"CAGE-CAL split overlap: {key}")
        if ids_digest(train | validation) != cell["fit_ids_hash"] or ids_digest(evaluation) != cell["eval_ids_hash"]:
            raise ValueError(f"CAGE-CAL fitting/evaluation split mismatch: {key}")
        selected = []
        for question in cell["matched_ids"]:
            prediction = predictions[(cell["task"], cell["models"], question)]
            if prediction["target"] != cell["selected_targets"][question]:
                raise ValueError(f"CAGE-CAL target mismatch: {key}, {question}")
            if prediction["correct"] not in (0, 1):
                raise ValueError(f"invalid CAGE-CAL correctness label: {key}, {question}")
            selected.append(prediction)
        probabilities = np.round(np.asarray([row[score] for row in selected], dtype=float), 12)
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(f"invalid CAGE-CAL confidence: {key}")
        correct = np.asarray([row["correct"] for row in selected])
        values = {metric: float(getattr(metrics, metric)(probabilities, correct))
                  for metric in FULL_METRICS if metric != "accuracy"}
        values["accuracy"] = float(correct.mean())
        if not math.isclose(values["accuracy"], cell["vote_accuracy"], abs_tol=1e-12):
            raise ValueError(f"CAGE-CAL voting accuracy mismatch: {key}")
        exported = saved_metrics[key, method]
        for metric, value in values.items():
            expected = float(exported[metric]) if exported[metric] else float("nan")
            if not (math.isnan(value) and math.isnan(expected)) and not math.isclose(value, expected, abs_tol=1e-12):
                raise ValueError(f"CAGE-CAL saved {metric} does not match predictions: {key}")
        row = dict(references[key], method=method, judge_approximate=False, judge_target_mismatches=0)
        for metric, value in values.items():
            value = value if math.isfinite(value) else None
            reference = row.get(f"reference_{metric}")
            row[metric] = value
            row[f"delta_{metric}"] = value - reference if value is not None and reference is not None else None
        appended.append(row)
    files = ("manifest.json", "predictions.jsonl", "calibration.json", "paper_tables/audit.json", "paper_tables/cagecal_metrics.csv")
    provenance = {"directory": str(directory.resolve()), "run_identity": manifest["identity"], "score": score,
                  "method": method, "seeds": manifest["config"]["seeds"], "n_cells": len(appended),
                  "calibration": calibration, "sources": {name: digest(directory / name) for name in files},
                  "prediction": "mean across training seeds, followed by BetaSB when available" if score == "betasb" else "mean across training seeds",
                  "spread": "sample SD across model-group deltas, not training seeds",
                  "same_prediction_across_estimators": "evaluated on each estimator's existing matched question mask"}
    return original + appended, {method: LABELS[method]}, provenance


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
    parser.add_argument("--score", choices=("raw", "betasb"), default="betasb")
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
    print(f"Added {provenance['n_cells']} CAGE-CAL cells; wrote {len(files)} tables to {directory}")


if __name__ == "__main__":
    main()
