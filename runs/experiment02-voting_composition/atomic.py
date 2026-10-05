"""One CSV row per (dataset, model group): each member's own-answer baseline beside the group's vote."""

import argparse
import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path

from conf_compose.composition import (candidate_set, from_zero_shot, majority_answer, pool_methods, support)
from conf_compose.composition.evidence import Item, Stream
from conf_compose.constants import RESULTS_DIR, TASKS
from conf_compose.data import Example, get_task
from conf_compose.pipelines.inference import STORE, InferenceSettings, inference_dir
from conf_compose.utils.metrics import auarc, auroc, brier, ece, nll

RULES = ("mean", "logodds_sum", "logodds_mean")
METRICS = ("ece", "auarc", "auroc", "brier", "nll")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True, choices=sorted(TASKS))
    parser.add_argument("--group", nargs="+", required=True, help="the models that vote in this group")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--estimator", choices=("cons", "seq"), default="cons")
    parser.add_argument("--context", choices=("direct", "reasoned"), default="direct")
    parser.add_argument("--seq-score", choices=("norm_sum", "norm_mean"), default="norm_sum")
    parser.add_argument("--match", default="", help="select saved cells by folder substring")
    parser.add_argument("--voter", type=int, default=0)
    parser.add_argument("--all-signals", action="store_true", default=True)
    parser.add_argument("--no-verbalized", action="store_true")
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument("--out", default=str(RESULTS_DIR / "voting_atomic" / "atomic.csv"))
    return parser.parse_args()


def cell_paths(task, args, split):
    """The group's inference cells; the flags must match the generation run or the digests will not resolve."""
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
        paths[inference_dir(settings, args.store).name] = path
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
    return {model: {row["id"]: row for row in
                    (json.loads(line) for line in path.read_text().splitlines() if line.strip())}
            for model, path in paths.items()}


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


def scored(values, labels, is_probability=True):
    """Every metric we can report for one score series, on the examples where it exists."""
    if len(values) < 10 or not 0 < sum(labels) < len(labels):
        return {m: None for m in METRICS}
    out = {"auroc": auroc(values, labels), "auarc": auarc(values, labels)}
    if is_probability:
        out.update({"ece": ece(values, labels), "brier": brier(values, labels), "nll": nll(values, labels)})
    else:
        out.update({"ece": None, "brier": None, "nll": None})
    return out


def run_task(task_name, args, items=None, records=None):
    task = get_task(task_name)
    if items is None:
        paths = cell_paths(task_name, args, "test")
        items = from_zero_shot(paths)
        records = load_records(paths) if args.estimator == "seq" else {}
    if not items:
        raise SystemExit(f"no shared examples for {task_name}")
    order = [s.model for s in sorted(items[0].round_streams((0,)), key=lambda s: s.agent)]

    own = {model: ([], []) for model in order}
    vote_labels, all_vote_labels, pooled = [], [], {rule: [] for rule in RULES}
    all_own = {model: [] for model in order}
    agree = 0
    for item in items:
        streams = item.round_streams((0,))
        example = Example(item.example_id, item.question, item.gold)
        candidates = candidate_set(task, item, streams)
        supports = {s.model: support(task, s, candidates, args.samples) for s in streams}

        def confidence(stream, target):
            if args.estimator == "seq":
                return sequence_score(task, records[stream.model][item.example_id], target, args)
            return supports[stream.model].binary(target)

        for stream in streams:
            all_own[stream.model].append(float(stream.answer is not None and task.is_correct(stream.answer, example)))
            if stream.answer is None:
                continue
            value = confidence(stream, stream.answer)
            if value is not None:
                own[stream.model][0].append(value)
                own[stream.model][1].append(float(task.is_correct(stream.answer, example)))

        target = majority_answer(task, streams)
        all_vote_labels.append(float(target is not None and task.is_correct(target, example)))
        if target is None:
            continue
        agree += all(s.answer is not None and task.equivalent(s.answer, target) for s in streams)
        scores = [confidence(s, target) for s in streams]
        scores = [v for v in scores if v is not None]
        if len(scores) < len(streams):
            continue
        vote_labels.append(float(task.is_correct(target, example)))
        for rule, prediction in pool_methods(scores).items():
            if rule in RULES:
                pooled[rule].append(prediction.score)

    row = {"task": task_name, "n_models": len(order), "models": "|".join(_short(m) for m in order),
           "estimator": args.estimator, "context": args.context if args.estimator == "seq" else "",
           "seq_score": args.seq_score if args.estimator == "seq" else "",
           "match": args.match, "voter": args.voter,
           "n_samples": args.samples, "n_examples": len(items), "n_voted": len(vote_labels),
           "unanimous": round(agree / len(items), 4) if items else None}
    row["single_acc"] = _list(sum(all_own[m]) / len(items) for m in order)
    row["single_coverage"] = _list(len(own[m][0]) / len(items) for m in order)
    row["single_scored_acc"] = _list(sum(l) / len(l) if l else None for _, l in (own[m] for m in order))
    for metric in ("ece", "auarc", "auroc"):
        row[f"single_{metric}"] = _list(scored(v, l)[metric] for v, l in (own[m] for m in order))
    row["vote_acc"] = _round(sum(all_vote_labels) / len(items))
    row["vote_coverage"] = _round(len(vote_labels) / len(items))
    row["vote_scored_acc"] = _round(sum(vote_labels) / len(vote_labels)) if vote_labels else None
    for rule in RULES:
        stats = scored(pooled[rule], vote_labels)
        for metric in ("ece", "auarc", "auroc", "brier", "nll"):
            row[f"{rule}_{metric}"] = _round(stats[metric])
    return row


def main():
    args = parse_args()
    if len(set(args.group)) != len(args.group):
        raise SystemExit("--group must contain distinct models")
    if args.samples < 1 or any(s < 1 or s > len(args.group) for s in (args.sizes or [])):
        raise SystemExit("invalid sample count or group size")
    if args.estimator == "seq" and not args.match:
        raise SystemExit("--estimator seq requires --match to select candidate-scored cells")
    rows = []
    for task_name in args.tasks:
        paths = cell_paths(task_name, args, "test")
        records = load_records(paths)
        groups = [tuple(paths)] if not args.sizes else [group for size in dict.fromkeys(args.sizes)
                                                       for group in combinations(paths, size)]
        for group in groups:
            items = group_items(records, group)
            rows.append(run_task(task_name, args, items, records))
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
    print(",".join(rows[0]))
    for row in rows:
        print(",".join("" if v is None else str(v) for v in row.values()))
    print(f"\nappended {len(rows)} row(s) to {out}", file=sys.stderr)


def _short(model):
    return model.split("/")[-1].split("--")[0].replace("vllm__", "")


def _round(value):
    return None if value is None else round(value, 4)


def _list(values):
    return "[" + " ".join("na" if v is None else f"{v:.4f}" for v in values) + "]"


if __name__ == "__main__":
    main()
