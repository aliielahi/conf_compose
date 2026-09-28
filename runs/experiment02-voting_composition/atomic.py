"""One CSV row per (dataset, model group): each member's own-answer baseline beside the group's vote."""

import argparse
import csv
import sys
from pathlib import Path

from conf_compose.composition import (candidate_set, from_zero_shot, majority_answer, pool_methods, support)
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


def run_task(task_name, args):
    task = get_task(task_name)
    items = from_zero_shot(cell_paths(task_name, args, "test"))
    order = [s.model for s in sorted(items[0].round_streams((0,)), key=lambda s: s.agent)]

    own = {model: ([], []) for model in order}
    vote_labels, pooled = [], {rule: [] for rule in RULES}
    agree = 0
    for item in items:
        streams = item.round_streams((0,))
        example = Example(item.example_id, item.question, item.gold)
        candidates = candidate_set(task, item, streams)
        supports = {s.model: support(task, s, candidates, args.samples) for s in streams}

        for stream in streams:
            if stream.answer is None:
                continue
            value = supports[stream.model].binary(stream.answer)
            if value is not None:
                own[stream.model][0].append(value)
                own[stream.model][1].append(float(task.is_correct(stream.answer, example)))

        target = majority_answer(task, streams)
        if target is None:
            continue
        agree += len({a for a in (s.answer for s in streams) if a is not None}) == 1
        scores = [supports[s.model].binary(target) for s in streams]
        scores = [v for v in scores if v is not None]
        if len(scores) < len(streams):
            continue
        vote_labels.append(float(task.is_correct(target, example)))
        for rule, prediction in pool_methods(scores).items():
            if rule in RULES:
                pooled[rule].append(prediction.score)

    row = {"task": task_name, "n_models": len(order), "models": "|".join(_short(m) for m in order),
           "n_samples": args.samples, "n_examples": len(items), "n_voted": len(vote_labels),
           "unanimous": round(agree / len(items), 4) if items else None}
    row["single_acc"] = _list(sum(l) / len(l) if l else None for _, l in (own[m] for m in order))
    for metric in ("ece", "auarc", "auroc"):
        row[f"single_{metric}"] = _list(scored(v, l)[metric] for v, l in (own[m] for m in order))
    row["vote_acc"] = round(sum(vote_labels) / len(vote_labels), 4) if vote_labels else None
    for rule in RULES:
        stats = scored(pooled[rule], vote_labels)
        for metric in ("ece", "auarc", "auroc", "brier", "nll"):
            row[f"{rule}_{metric}"] = _round(stats[metric])
    return row


def main():
    args = parse_args()
    rows = [run_task(task, args) for task in args.tasks]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
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
