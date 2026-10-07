import argparse
import csv
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from conf_compose.composition import (candidate_set, majority_answer, pool_methods, support)
from conf_compose.composition.evidence import Item, Stream
from conf_compose.composition.pooling import FIXED_RULES, RULES, fit_methods
from conf_compose.constants import RESULTS_DIR, TASKS
from conf_compose.data import Example, get_task
from conf_compose.pipelines.inference import STORE, InferenceSettings, inference_dir
from conf_compose.utils.metrics import auarc, auroc, brier, ece, nll

METRICS = ("ece", "auarc", "auroc", "brier", "nll")


def parse_args():
    parser = argparse.ArgumentParser(description="Atomic voting confidence with fixed and fitted pooling rules")
    parser.add_argument("--tasks", nargs="+", required=True, choices=sorted(TASKS))
    parser.add_argument("--group", nargs="+", required=True, help="the models that vote in this group")
    add_common_arguments(parser)
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--estimator", choices=("cons", "seq"), default="cons")
    parser.add_argument("--out", default=str(RESULTS_DIR / "voting_atomic" / "atomic_learned.csv"))
    return parser.parse_args()


def add_common_arguments(parser):
    parser.add_argument("--fit-fraction", type=float, default=0.3, help="fraction of saved test questions reserved for fitting")
    parser.add_argument("--fit-seed", type=int, default=0)
    parser.add_argument("--fit-split", choices=("holdout", "validation"), default="holdout")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--tie-break", choices=("first", "confidence"), default="confidence")
    parser.add_argument("--tie-seed", type=int, default=0)
    parser.add_argument("--context", choices=("direct", "reasoned"), default="direct")
    parser.add_argument("--seq-score", choices=("norm_sum", "norm_mean"), default="norm_sum")
    parser.add_argument("--match", default="", help="select saved cells by folder substring")
    parser.add_argument("--voter", type=int, default=0)
    parser.add_argument("--all-signals", action="store_true", default=True)
    parser.add_argument("--no-verbalized", action="store_true")
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument("--ablations", action="store_true", help="also fit equal-weight BLP and diagonal Kahn")
    parser.add_argument("--logistic-l2", type=float, default=1.0)


def cell_paths(task, args, split):
    paths = {}
    for model in args.group:
        if args.match:
            tag = model.replace("/", "__")
            matches = [p for p in Path(args.store, task).glob(f"{tag}--*/{split}.jsonl")
                       if args.match in p.parent.name
                       and f"s70v{args.voter}_k{args.samples}_" in p.parent.name]
            if len(matches) != 1:
                raise SystemExit(f"expected one cell for {task}/{model}, found {len(matches)}: {matches}")
            paths[model] = matches[0]
            continue
        settings = InferenceSettings(task=task, model=model, n_val=0, n_test=-1,
                                     answer_temperature=0.7, voter=args.voter,
                                     consistency_samples=args.samples,
                                     verbalized=not args.no_verbalized,
                                     verification_context=args.all_signals, debias=args.all_signals)
        path = inference_dir(settings, args.store) / f"{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing inference cell:\n  {path}\n"
                             f"check --samples/--voter/--no-verbalized match how it was generated")
        paths[model] = path
    return paths


def sequence_score(task, record, target, args):
    candidates = (record.get("candidate_scores") or {}).get("candidates") or []
    if target is None or not candidates:
        return None
    scores = []
    selected = []
    for candidate in candidates:
        logs = (candidate.get(args.context) or {}).get("logprobs")
        if not logs or not all(math.isfinite(v) for v in logs):
            return None
        value = sum(logs)
        if args.seq_score == "norm_mean":
            value /= len(logs)
        scores.append(value)
        selected.append(task.equivalent(candidate["answer"], target))
    if not any(selected):
        return None
    top = max(scores)
    weights = [math.exp(value - top) for value in scores]
    return sum(w for w, match in zip(weights, selected) if match) / sum(weights)


def load_records(paths):
    records = {}
    for model, path in paths.items():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        records[model] = {row["id"]: row for row in rows}
        if len(records[model]) != len(rows):
            raise ValueError(f"duplicate question IDs in {path}")
    return records


def validation_records(paths):
    validation_paths = {model: path.with_name("validation.jsonl") for model, path in paths.items()}
    missing = [str(path) for path in validation_paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"validation records missing from selected inference cells: {missing}")
    return load_records(validation_paths)


def group_items(records, group):
    shared = sorted(set.intersection(*(set(records[model]) for model in group)))
    items = []
    for example_id in shared:
        first = records[group[0]][example_id]
        streams = []
        for agent, model in enumerate(group):
            row = records[model][example_id]
            if row["gold"] != first["gold"]:
                raise ValueError(f"inconsistent gold for {example_id}")
            streams.append(Stream(f"{example_id}:r0:a{agent}", agent, 0, model, row["prediction"],
                                  row.get("sampled_answers", {}).get("consistency_t0.7", [])))
        items.append(Item(example_id, first.get("question", ""), first["gold"], streams, first.get("options")))
    return items


def scored(values, labels, ranking=None):
    if not values:
        return {metric: None for metric in METRICS}
    ranking = values if ranking is None else ranking
    result = {"auroc": auroc(ranking, labels), "auarc": auarc(ranking, labels),
              "ece": ece(values, labels), "brier": brier(values, labels), "nll": nll(values, labels)}
    return {key: value if math.isfinite(value) else None for key, value in result.items()}


def split_items(task_name, items, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("--fit-fraction must be between zero and one")
    fitting, evaluation = [], []
    for item in items:
        key = json.dumps([task_name, seed, item.example_id]).encode()
        position = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        (fitting if position < fraction else evaluation).append(item)
    if not fitting or not evaluation:
        raise ValueError("the holdout needs fitting and evaluation questions; change --fit-fraction or supply validation")
    return fitting, evaluation


def confidence_scores(task, item, args, records):
    streams = item.round_streams((0,))
    candidates = candidate_set(task, item, streams)
    supports = {stream.model: support(task, stream, candidates, args.samples) for stream in streams}

    def confidence(stream, target):
        if args.estimator == "seq":
            return sequence_score(task, records[stream.model][item.example_id], target, args)
        if len(stream.samples) < args.samples:
            return None
        return supports[stream.model].binary(target)

    target, _ = selected_answer(task, item, args, supports)
    own = [confidence(stream, stream.answer) for stream in streams]
    shared = [confidence(stream, target) for stream in streams]
    return target, own, shared


def selected_answer(task, item, args, supports=None):
    streams = item.round_streams((0,))
    candidates = candidate_set(task, item, streams)
    tie_break = getattr(args, "tie_break", "first")
    if tie_break not in ("first", "confidence"):
        raise ValueError(f"unknown tie-break rule: {tie_break}")
    if not candidates:
        return None, "no_answer"
    counts = {candidate: sum(stream.answer is not None and task.equivalent(stream.answer, candidate)
                             for stream in streams) for candidate in candidates}
    tied = [candidate for candidate in candidates if counts[candidate] == max(counts.values())]
    if tie_break == "first":
        return majority_answer(task, streams), "first" if len(tied) > 1 else "count"
    supports = supports or {stream.model: support(task, stream, candidates, args.samples) for stream in streams}
    weights = {stream.stream_id: supports[stream.model].binary(stream.answer)
               if len(stream.samples) >= args.samples else None for stream in streams}
    unavailable = any(weights[stream.stream_id] is None for stream in streams
                      if stream.answer is not None and any(task.equivalent(stream.answer, candidate) for candidate in tied))
    tie_key = json.dumps([getattr(args, "tie_seed", 0), task.name, item.example_id,
                          sorted(stream.model for stream in streams)])
    reason = "count"
    if len(tied) > 1:
        reason = "missing_consistency_seeded" if unavailable else "confidence"
        if not unavailable:
            totals = [round(sum(weights[stream.stream_id] for stream in streams
                                if stream.answer is not None and task.equivalent(stream.answer, candidate)), 12)
                      for candidate in tied]
            if totals.count(max(totals)) > 1:
                reason = "equal_confidence_seeded"
    return majority_answer(task, streams, None if unavailable else weights, tie_key=tie_key), reason


def id_digest(ids):
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def run_task(task_name, args, items=None, records=None, fitting=None, fitting_records=None):
    task = get_task(task_name)
    if items is None:
        paths = cell_paths(task_name, args, "test")
        records = load_records(paths)
        items = group_items(records, tuple(records))
        if args.fit_split == "validation" and fitting is None:
            fitting_records = validation_records(paths)
            fitting = group_items(fitting_records, tuple(records))
    if not items:
        raise ValueError(f"no shared examples for {task_name}")
    n_total = len(items)
    if fitting is None:
        if args.fit_split != "holdout":
            raise ValueError("validation fitting requires separate validation records")
        fitting, items = split_items(task_name, items, args.fit_fraction, args.fit_seed)
        fitting_records = records
    if not fitting:
        raise ValueError("no shared fitting questions")
    fitting_records = records if fitting_records is None else fitting_records
    fit_ids = {item.example_id for item in fitting}
    eval_ids = {item.example_id for item in items}
    if fit_ids & eval_ids:
        raise ValueError("fitting and evaluation question IDs overlap")
    order = [stream.model for stream in items[0].round_streams((0,))]
    for item in [*fitting, *items]:
        if [stream.model for stream in item.round_streams((0,))] != order:
            raise ValueError("fitting and evaluation must have the same ordered streams")

    fit_scores, fit_labels, scored_fit_ids = [], [], []
    for item in fitting:
        target, _, scores = confidence_scores(task, item, args, fitting_records)
        if target is not None and all(value is not None for value in scores):
            fit_scores.append(scores)
            fit_labels.append(float(task.is_correct(target, Example(item.example_id, item.question, item.gold))))
            scored_fit_ids.append(item.example_id)
    matrix = np.asarray(fit_scores, dtype=float).reshape(-1, len(order))
    fitted = fit_methods(matrix, fit_labels, ablations=getattr(args, "ablations", False),
                         logistic_l2=getattr(args, "logistic_l2", 1.0))
    rules = (*FIXED_RULES, *fitted)

    own = {model: ([], []) for model in order}
    all_own = {model: [] for model in order}
    vote_labels, all_vote_labels, scored_eval_ids = [], [], []
    pooled = {rule: [] for rule in rules}
    ranks = {rule: [] for rule in rules}
    agree = 0
    for item in items:
        streams = item.round_streams((0,))
        example = Example(item.example_id, item.question, item.gold)
        target, own_scores, scores = confidence_scores(task, item, args, records)
        for stream, value in zip(streams, own_scores):
            correct = float(stream.answer is not None and task.is_correct(stream.answer, example))
            all_own[stream.model].append(correct)
            if value is not None:
                own[stream.model][0].append(value)
                own[stream.model][1].append(correct)

        correct = float(target is not None and task.is_correct(target, example))
        all_vote_labels.append(correct)
        if target is None:
            continue
        agree += all(stream.answer is not None and task.equivalent(stream.answer, target) for stream in streams)
        if any(value is None for value in scores):
            continue
        vote_labels.append(correct)
        scored_eval_ids.append(item.example_id)
        predictions = pool_methods(scores)
        predictions.update({name: fit.predict(scores) for name, fit in fitted.items()})
        for rule, prediction in predictions.items():
            if prediction.score is not None:
                pooled[rule].append(prediction.score)
                rank = prediction.ranking_score
                if rank is None:
                    rank = prediction.logit if prediction.logit is not None else prediction.score
                ranks[rule].append(rank)

    row = {"task": task_name, "n_models": len(order), "models": "|".join(_short(model) for model in order),
           "model_ids": json.dumps(order), "estimator": args.estimator,
           "context": args.context if args.estimator == "seq" else "",
           "seq_score": args.seq_score if args.estimator == "seq" else "",
           "match": args.match, "voter": args.voter, "n_samples": args.samples,
           "n_total": n_total, "n_examples": len(items), "n_voted": len(vote_labels),
           "tie_break": getattr(args, "tie_break", "first"), "tie_seed": getattr(args, "tie_seed", 0),
           "selection_signal": "consistency_t0.7", "selection_smoothing": "add_half",
           "logistic_l2": getattr(args, "logistic_l2", 1.0),
           "fit_split": args.fit_split,
           "fit_fraction": args.fit_fraction if args.fit_split == "holdout" else None,
           "fit_seed": args.fit_seed if args.fit_split == "holdout" else None,
           "n_fit_questions": len(fitting), "n_fit_scored": len(fit_labels),
           "n_fit_errors": len(fit_labels) - int(sum(fit_labels)),
           "fit_ids_hash": id_digest(fit_ids), "eval_ids_hash": id_digest(eval_ids),
           "fit_scored_ids_hash": id_digest(scored_fit_ids), "eval_scored_ids_hash": id_digest(scored_eval_ids),
           "unanimous": _round(agree / len(items))}
    row["single_acc"] = _list(sum(all_own[model]) / len(items) for model in order)
    row["single_coverage"] = _list(len(own[model][0]) / len(items) for model in order)
    row["single_scored_acc"] = _list(sum(labels) / len(labels) if labels else None for _, labels in own.values())
    for metric in METRICS:
        row[f"single_{metric}"] = _list(scored(values, labels)[metric] for values, labels in own.values())
    row["vote_acc"] = _round(sum(all_vote_labels) / len(items))
    row["vote_coverage"] = _round(len(vote_labels) / len(items))
    row["vote_scored_acc"] = _round(sum(vote_labels) / len(vote_labels)) if vote_labels else None
    for rule in rules:
        stats = scored(pooled[rule], vote_labels, ranks[rule])
        for metric in METRICS:
            row[f"{rule}_{metric}"] = _round(stats[metric])
        row[f"{rule}_coverage"] = _round(len(pooled[rule]) / len(items))
    for name, fit in fitted.items():
        row[f"{name}_status"] = fit.status
        row[f"{name}_scale"] = fit.scale
        row[f"{name}_parameters"] = json.dumps(fit.parameters, sort_keys=True)
    return row


def main():
    args = parse_args()
    if len(set(args.group)) != len(args.group):
        raise SystemExit("--group must contain distinct models")
    if args.samples < 1 or any(s < 1 or s > len(args.group) for s in (args.sizes or [])):
        raise SystemExit("invalid sample count or group size")
    if args.estimator == "seq" and not args.match:
        raise SystemExit("--estimator seq requires --match to select candidate-scored cells")
    if not 0 < args.fit_fraction < 1:
        raise SystemExit("--fit-fraction must be between zero and one")
    rows = []
    for task_name in args.tasks:
        paths = cell_paths(task_name, args, "test")
        records = load_records(paths)
        fitting_records = validation_records(paths) if args.fit_split == "validation" else None
        groups = [tuple(paths)] if not args.sizes else [group for size in dict.fromkeys(args.sizes)
                                                       for group in combinations(paths, size)]
        for group in groups:
            items = group_items(records, group)
            fitting = group_items(fitting_records, group) if fitting_records is not None else None
            if fitting is not None and not fitting:
                raise SystemExit(f"no shared validation questions for {task_name}/{group}")
            rows.append(run_task(task_name, args, items, records, fitting, fitting_records))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    if exists and out.stat().st_size:
        with out.open(newline="") as handle:
            if next(csv.reader(handle)) != list(rows[0]):
                raise SystemExit(f"CSV columns differ; choose a new --out path: {out}")
    else:
        exists = False
    with out.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    print(f"\nappended {len(rows)} row(s) to {out}", file=sys.stderr)


def _short(model):
    return model.split("/")[-1].split("--")[0].replace("vllm__", "")


def _round(value):
    return None if value is None else round(value, 4)


def _list(values):
    return "[" + " ".join("na" if v is None else f"{v:.4f}" for v in values) + "]"


if __name__ == "__main__":
    main()
