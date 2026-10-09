import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from conf_compose.composition.pooling import FittedBLP, FittedPool, pool_methods
from conf_compose.constants import ROOT
from conf_compose.data import Example, get_task
from conf_compose.utils.calibration import fit_temperature, temperature_scale
from conf_compose.utils.metrics import auarc, auroc, brier, ece, nll

METRICS = ("accuracy", "ece", "auarc", "auroc", "brier", "nll")
REPORT_METRICS = (*METRICS, "t_brier", "t_ece")
LOWER_IS_BETTER = {"ece", "brier", "nll"}
METHODS = {
    "mean": "Arithmetic mean",
    "logodds_sum": "Log-odds sum",
    "logodds_mean": "Log-odds mean",
    "shared_rho": "Shared rho",
    "shared_scale": "Shared scale",
    "kahn": "Kahn (full covariance)",
    "blp": "Weighted BLP",
    "logistic_pool": "Regularized logistic pooling",
}
ABLATIONS = {"blp_equal": "Equal-weight BLP", "kahn_diagonal": "Kahn (diagonal covariance)"}
SIGNALS = {"cons": "consistency_t0.7", "seq": "cand_direct_sum"}


def atomic_module():
    path = ROOT / "runs/experiment02-voting_composition/atomic.py"
    spec = importlib.util.spec_from_file_location("voting_atomic_report_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path):
    rows = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["id"] in rows:
                    raise ValueError(f"duplicate ID in {path}: {row['id']}")
                rows[row["id"]] = row
    return rows


def load_csvs(directory, estimators):
    cells = []
    for estimator in estimators:
        with (directory / f"{estimator}.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        seen = set()
        for row in rows:
            key = (row["task"], row["models"])
            if key in seen or row["estimator"] != estimator:
                raise ValueError(f"duplicate or incorrectly labeled result: {key}, {estimator}")
            seen.add(key)
            cells.append(row)
    return cells


def index_judges(directory, judges):
    if not directory.is_dir():
        raise FileNotFoundError(f"judge directory does not exist: {directory}; set --judge-dir to the downloaded results")
    result = {}
    for path in sorted(directory.glob("*/*/manifest.json")):
        manifest = json.loads(path.read_text())
        args = manifest["args"]
        judge = args["judge"].split("/")[-1]
        if judge not in judges or args.get("split", "test") != "test":
            continue
        view = args["view"]
        signal = args["confidence_method"] if view == "reasoning_confidence" else "none"
        if view not in ("reasoning", "reasoning_confidence") or signal not in ("none", *SIGNALS.values()):
            continue
        panel = tuple(model.split("/")[-1] for model in args["panel"])
        key = (args["task"], panel, judge, view, signal, int(args.get("voter", 0)))
        if key in result:
            raise ValueError(f"ambiguous judge cells: {result[key]} and {path}")
        if path.with_name("verdicts.jsonl").exists() and path.with_name("metrics.json").exists():
            result[key] = path
    found = {key[2] for key in result}
    absent = set(judges) - found
    if absent:
        raise ValueError(f"no completed judge cells for {sorted(absent)} in {directory}")
    return result


def restore_fit(row, method):
    parameters = json.loads(row[f"{method}_parameters"])
    scale = row[f"{method}_scale"]
    fields = dict(name=method, n_streams=int(row["n_models"]), scale=float(scale) if scale else None,
                  status=row[f"{method}_status"], n_fit=int(row["n_fit_scored"]),
                  n_errors=int(row["n_fit_errors"]), parameters=parameters,
                  weights=tuple(parameters["weights"]) if "weights" in parameters else None,
                  intercept=float(parameters.get("intercept", 0)))
    if method.startswith("blp"):
        return FittedBLP(**fields, alpha=parameters.get("alpha", 1), beta=parameters.get("beta", 1))
    return FittedPool(**fields)


def rank_value(prediction):
    if prediction.ranking_score is not None:
        return prediction.ranking_score
    return prediction.logit if prediction.logit is not None else prediction.score


def valid_probability(value):
    return value is not None and math.isfinite(value) and 0 <= value <= 1


def metric_values(scores, correct, ranks=None):
    probabilities = np.round(np.asarray(scores, dtype=float), 12)
    ranking = np.round(np.asarray(scores if ranks is None else ranks, dtype=float), 12)
    values = {"accuracy": float(np.mean(correct)), "ece": float(ece(probabilities, correct)),
              "auarc": float(auarc(ranking, correct)), "auroc": float(auroc(ranking, correct)),
              "brier": float(brier(probabilities, correct)), "nll": float(nll(probabilities, correct))}
    return {key: value if math.isfinite(value) else None for key, value in values.items()}


def reference_indices(mode, fitting_accuracy, solo_metrics, solo_accuracy, fitting_metrics=None, fitting_target_auarc=None):
    if mode == "fit_metric":
        if not fitting_metrics or any(value is None for value in fitting_metrics):
            raise ValueError("best fitting-metric reference requires scored fitting questions for every member")
        solo_metrics = fitting_metrics
    available = set(solo_metrics[0])
    if mode in ("metric_best", "fit_metric"):
        indices = {}
        for metric in METRICS:
            if metric == "accuracy":
                accuracy = fitting_accuracy if mode == "fit_metric" else solo_accuracy
                indices[metric] = max(range(len(accuracy)), key=lambda index: accuracy[index])
            elif metric in available:
                candidates = [index for index, values in enumerate(solo_metrics) if values[metric] is not None]
                if not candidates:
                    indices[metric] = 0
                else:
                    choose = min if metric in LOWER_IS_BETTER else max
                    indices[metric] = choose(candidates, key=lambda index: solo_metrics[index][metric])
        return indices
    if mode == "fit_auarc":
        if fitting_target_auarc is None or any(value is None for value in fitting_target_auarc):
            raise ValueError("fitting answer-matched AUARC is unavailable for a panel member")
        selected = max(range(len(fitting_target_auarc)), key=lambda index: fitting_target_auarc[index])
    else:
        accuracy = fitting_accuracy if mode == "fit_accuracy" else solo_accuracy
        selected = max(range(len(accuracy)), key=lambda index: accuracy[index])
    return dict.fromkeys(METRICS, selected)



class SourceStore:
    def __init__(self, directory):
        self.directory = directory
        self.task = None
        self.records = {}
        self.hashes = {}
        self.paths = {}

    def get(self, atomic, row, group, split):
        if self.task != row["task"]:
            self.task = row["task"]
            self.records.clear()
        args = row_arguments(row, self.directory)
        args.group = group
        paths = atomic.cell_paths(row["task"], args, split)
        expected = json.loads(row["input_hashes"])
        for model, path in paths.items():
            if path not in self.hashes:
                self.hashes[path] = digest(path)
            if self.hashes[path] != expected[model][split]:
                raise ValueError(f"inference changed since pooling fit: {path}")
            if path not in self.records:
                self.records[path] = read_jsonl(path)
            self.paths[str(path)] = self.hashes[path]
        return {model: self.records[path] for model, path in paths.items()}


def row_arguments(row, store):
    return SimpleNamespace(group=json.loads(row["model_ids"]), store=str(store), match=row["match"],
                           voter=int(row["voter"]), samples=int(row["n_samples"]),
                           estimator=row["estimator"], context=row["context"] or "direct",
                           seq_score=row["seq_score"] or "norm_sum", all_signals=True,
                           no_verbalized=False, tie_break=row.get("tie_break", "first"),
                           tie_seed=int(row.get("tie_seed", 0)))


def process_cell(row, sources, judge_index, judges, methods, reference, judge_policy="strict"):
    atomic = atomic_module()
    group = json.loads(row["model_ids"])
    args = row_arguments(row, sources.directory)
    task = get_task(row["task"])
    records = sources.get(atomic, row, group, "test")
    items = atomic.group_items(records, group)
    if row["fit_split"] == "holdout":
        fitting, evaluation = atomic.split_items(row["task"], items, float(row["fit_fraction"]), int(row["fit_seed"]))
        fitting_records = records
    else:
        fitting_records = sources.get(atomic, row, group, "validation")
        fitting = atomic.group_items(fitting_records, group)
        evaluation = items
    for name, subset in (("fit", fitting), ("eval", evaluation)):
        if atomic.id_digest([item.example_id for item in subset]) != row[f"{name}_ids_hash"]:
            raise ValueError(f"{name} IDs do not match saved pooling run for {row['task']}/{row['models']}")
    fitting_accuracy = [sum(bool(fitting_records[model][item.example_id]["correct"]) for item in fitting) / len(fitting)
                        for model in group]
    fitting_scores = [[] for _ in group]
    fitting_labels = [[] for _ in group]
    fitting_target_scores = [[] for _ in group]
    fitting_target_labels = []
    for item in fitting:
        target, own, shared_scores = atomic.confidence_scores(task, item, args, fitting_records)
        if target is not None and all(valid_probability(score) for score in shared_scores):
            fitting_target_labels.append(float(task.is_correct(target, Example(item.example_id, item.question, item.gold))))
            for index, score in enumerate(shared_scores):
                fitting_target_scores[index].append(score)
        if all(valid_probability(score) for score in own):
            for index, model in enumerate(group):
                fitting_scores[index].append(own[index])
                fitting_labels[index].append(float(fitting_records[model][item.example_id]["correct"]))
    fitting_metrics = [metric_values(scores, labels) if scores else None
                       for scores, labels in zip(fitting_scores, fitting_labels)]
    fitting_target_auarc = [metric_values(scores, fitting_target_labels)["auarc"] if scores else None
                            for scores in fitting_target_scores]
    judge_rows, missing, judge_sources = {}, {}, {}
    panel = tuple(model.split("/")[-1] for model in group)
    for judge in judges:
        for view in ("reasoning", "reasoning_confidence"):
            method = f"judge:{judge}:{view}"
            signal = SIGNALS[row["estimator"]] if view == "reasoning_confidence" else "none"
            key = (row["task"], panel, judge, view, signal, int(row["voter"]))
            path = judge_index.get(key)
            if path is None:
                missing[method] = "judge cell absent"
                continue
            manifest = json.loads(path.read_text())
            if manifest["args"].get("match", "") != row["match"]:
                raise ValueError(f"judge used a different inference selector: {path}")
            judge_rows[method] = read_jsonl(path.with_name("verdicts.jsonl"))
            judge_sources[method] = {"path": str(path.parent),
                                     "verdicts_hash": digest(path.with_name("verdicts.jsonl")),
                                     "context": json.loads(path.with_name("metrics.json").read_text()).get("context")}
    fitted = {method: restore_fit(row, method) for method in methods if method not in ("mean", "logodds_sum", "logodds_mean")}
    unavailable = {name for name, fit in fitted.items() if fit.weights is None and fit.scale is None}
    missing.update({name: f"fit status: {fitted[name].status}" for name in unavailable})
    active_methods = [method for method in methods if method not in unavailable]
    values = {method: [] for method in [*active_methods, *judge_rows]}
    rankings = {method: [] for method in values}
    solo = [[] for _ in group]
    solo_labels = [[] for _ in group]
    shared = [[] for _ in group]
    labels, retained_ids, scored_ids = [], [], []
    judge_mismatches = {method: [] for method in judge_rows}
    old_targets, new_targets, selection_reasons = {}, {}, {}
    exclusions = {"source": 0, "judge": 0}
    replay_values = {method: [] for method in active_methods}
    replay_ranks = {method: [] for method in active_methods}
    replay_labels = []
    for item in evaluation:
        target, own, scores = atomic.confidence_scores(task, item, args, records)
        legacy_args = SimpleNamespace(**{**vars(args), "tie_break": "first"})
        old_targets[item.example_id] = atomic.confidence_scores(task, item, legacy_args, records)[0]
        new_targets[item.example_id] = target
        _, selection_reasons[item.example_id] = atomic.selected_answer(task, item, args)
        original_label = None
        if target is not None and all(valid_probability(score) for score in scores):
            original_label = float(task.is_correct(target, Example(item.example_id, item.question, item.gold)))
            scored_ids.append(item.example_id)
            predictions = pool_methods(scores)
            predictions.update({method: fit.predict(scores) for method, fit in fitted.items() if method not in unavailable})
            replay_labels.append(original_label)
            for method in active_methods:
                replay_values[method].append(predictions[method].score)
                replay_ranks[method].append(rank_value(predictions[method]))
        if original_label is None or not all(valid_probability(score) for score in own):
            exclusions["source"] += 1
            continue
        selected_judges = {}
        for method, verdicts in judge_rows.items():
            verdict = verdicts.get(item.example_id)
            if verdict is None:
                continue
            if not task.equivalent(verdict["final_answer"], target):
                if judge_policy == "strict":
                    raise ValueError(f"judge answer/label differs from majority: {method}/{item.example_id}")
                if valid_probability(verdict.get("verbalized")):
                    judge_mismatches[method].append(item.example_id)
            elif bool(verdict["correct"]) != bool(original_label):
                raise ValueError(f"judge correctness label differs for the same answer: {method}/{item.example_id}")
            if valid_probability(verdict.get("verbalized")):
                selected_judges[method] = float(verdict["verbalized"])
        if len(selected_judges) != len(judge_rows):
            exclusions["judge"] += 1
            continue
        retained_ids.append(item.example_id)
        labels.append(original_label)
        for index, model in enumerate(group):
            solo[index].append(own[index])
            solo_labels[index].append(float(records[model][item.example_id]["correct"]))
            shared[index].append(scores[index])
        for method in active_methods:
            values[method].append(predictions[method].score)
            rankings[method].append(rank_value(predictions[method]))
        for method, value in selected_judges.items():
            values[method].append(value)
            rankings[method].append(value)
    if atomic.id_digest(scored_ids) != row["eval_scored_ids_hash"]:
        raise ValueError("scored question IDs differ from saved pooling results")
    replay_differences = {}
    for method in active_methods:
        if not replay_values[method]:
            continue
        replay = metric_values(replay_values[method], replay_labels, replay_ranks[method])
        replay_differences[method] = {metric: value - float(row[f"{method}_{metric}"])
                                      for metric, value in replay.items() if metric != "accuracy"
                                      and value is not None and row.get(f"{method}_{metric}") not in (None, "")}
    expected_accuracy = float(row["vote_scored_acc"])
    if replay_labels and abs(float(np.mean(replay_labels)) - expected_accuracy) > 0.000051:
        raise ValueError("reconstructed vote accuracy differs from the saved run")
    metadata = {"task": row["task"], "estimator": row["estimator"], "models": row["models"],
                "n_models": len(group), "n_samples": int(row["n_samples"]),
                "n_evaluation": len(evaluation), "n_matched": len(labels),
                "coverage": len(labels) / len(evaluation), "exclusions": exclusions,
                "fit_ids_hash": row["fit_ids_hash"], "eval_ids_hash": row["eval_ids_hash"],
                "matched_ids_hash": atomic.id_digest(retained_ids), "matched_ids": retained_ids,
                "judge_sources": judge_sources, "missing": missing, "source_code_hash": row["code_hash"],
                "replay_minus_csv": replay_differences, "numerical_precision": 12,
                "fitting_accuracy": dict(zip(group, fitting_accuracy)),
                "fitting_solo_metrics": dict(zip(group, fitting_metrics)),
                "fitting_target_auarc": dict(zip(group, fitting_target_auarc)), "reference_mode": reference,
                "tie_break": args.tie_break, "tie_seed": args.tie_seed, "judge_policy": judge_policy,
                "old_targets": old_targets, "selected_targets": new_targets, "selection_reasons": selection_reasons,
                "n_changed_targets": sum(not task.equivalent(old_targets[key], value) for key, value in new_targets.items()),
                "judge_target_mismatch_ids": {method: [key for key in ids if key in set(retained_ids)]
                                              for method, ids in judge_mismatches.items()}}
    if not labels:
        metadata["missing"].update({method: "no common scored questions" for method in values})
        return [], metadata
    solo_metrics = [metric_values(scores, correct) for scores, correct in zip(solo, solo_labels)]
    solo_accuracy = [float(np.mean(correct)) for correct in solo_labels]
    indices = reference_indices(reference, fitting_accuracy, solo_metrics, solo_accuracy, fitting_metrics, fitting_target_auarc)
    baseline = {metric: solo_metrics[indices[metric]][metric] for metric in METRICS}
    matched_auarc = metric_values(shared[indices["accuracy"]], labels)["auarc"]
    fit_values = {method: [] for method in values}
    fit_labels = {method: [] for method in values}
    for item in fitting:
        target, _, scores = atomic.confidence_scores(task, item, args, fitting_records)
        if target is None or not all(valid_probability(score) for score in scores):
            continue
        label = float(task.is_correct(target, Example(item.example_id, item.question, item.gold)))
        predictions = pool_methods(scores)
        predictions.update({method: fit.predict(scores) for method, fit in fitted.items() if method not in unavailable})
        for method in active_methods:
            probability = predictions[method].score
            if valid_probability(probability):
                fit_values[method].append(probability)
                fit_labels[method].append(label)
        for method, verdicts in judge_rows.items():
            verdict = verdicts.get(item.example_id)
            if verdict is not None and valid_probability(verdict.get("verbalized")):
                fit_values[method].append(float(verdict["verbalized"]))
                fit_labels[method].append(label)
    solo_temperatures = [fit_temperature(scores, labels) if scores else None
                         for scores, labels in zip(fitting_scores, fitting_labels)]
    temperatures = {method: fit_temperature(fit_values[method], fit_labels[method]) if fit_values[method] else None
                    for method in values}
    solo_t = []
    for index, temperature in enumerate(solo_temperatures):
        scaled = temperature_scale(solo[index], temperature) if temperature is not None else None
        solo_t.append({"t_brier": float(brier(scaled, solo_labels[index])) if scaled is not None else None,
                       "t_ece": float(ece(scaled, solo_labels[index])) if scaled is not None else None})
    baseline["t_brier"] = solo_t[indices["brier"]]["t_brier"]
    baseline["t_ece"] = solo_t[indices["ece"]]["t_ece"]
    metadata["output_temperatures"] = temperatures
    metadata["solo_output_temperatures"] = dict(zip(group, solo_temperatures))
    metadata["reference_models"] = {metric: group[indices[metric]] for metric in METRICS}
    metadata["solo_metrics"] = {model: {**solo_metrics[index], "accuracy": solo_accuracy[index]} for index, model in enumerate(group)}
    metadata["vote_accuracy"] = float(np.mean(labels))
    output = []
    for method in ["best_solo", *values]:
        scores = baseline if method == "best_solo" else metric_values(values[method], labels, rankings[method])
        if method != "best_solo":
            temperature = temperatures[method]
            scaled = temperature_scale(values[method], temperature) if temperature is not None else None
            scores["t_brier"] = float(brier(scaled, labels)) if scaled is not None else None
            scores["t_ece"] = float(ece(scaled, labels)) if scaled is not None else None
        entry = {key: metadata[key] for key in ("task", "estimator", "models", "n_models", "n_matched", "coverage", "reference_mode")}
        entry.update(method=method, tie_break=args.tie_break, tie_seed=args.tie_seed,
                     output_temperature=(solo_temperatures[indices["brier"]] if method == "best_solo"
                                         else temperatures[method]),
                     judge_approximate=method.startswith("judge:") and judge_policy == "approximate",
                     judge_target_mismatches=len(metadata["judge_target_mismatch_ids"].get(method, [])))
        for metric in REPORT_METRICS:
            entry[metric] = scores[metric]
            entry[f"reference_{metric}"] = baseline[metric]
            entry[f"delta_{metric}"] = (scores[metric] - baseline[metric]
                                         if scores[metric] is not None and baseline[metric] is not None else None)
            reference_metric = metric[2:] if metric.startswith("t_") else metric
            entry[f"reference_{metric}_model"] = group[indices[reference_metric]]
            entry[f"reference_{metric}_accuracy"] = solo_accuracy[indices[reference_metric]]
        entry["answer_matched_auarc"] = matched_auarc if method == "best_solo" else scores["auarc"]
        entry["reference_answer_matched_auarc"] = matched_auarc
        entry["delta_answer_matched_auarc"] = (entry["answer_matched_auarc"] - matched_auarc
                                                 if entry["answer_matched_auarc"] is not None and matched_auarc is not None else None)
        output.append(entry)
    return output, metadata
