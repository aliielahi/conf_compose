"""Score every sequence-probability variant derivable from saved candidate log-probs, one model at a time."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from conf_compose.composition import base_model
from conf_compose.data import Example, get_task
from conf_compose.pipelines.inference import STORE
from conf_compose.utils.metrics import auarc, auroc, ece

# Every variant is a downstream choice over the same raw log-probs; probabilities also get ECE.
RAW = ("sum", "mean", "min", "tail10")
PROBABILITY = ("norm_sum", "norm_mean", "debiased")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--target", default="majority", choices=["majority", "own"])
    parser.add_argument("--match", default="", help="substring selecting one cell per model, e.g. _full_")
    parser.add_argument("--store", default=str(STORE))
    return parser.parse_args()


def aggregate(logprobs, kind):
    if not logprobs:
        return None
    if kind == "sum":
        return sum(logprobs)
    if kind == "mean":
        return sum(logprobs) / len(logprobs)
    if kind == "min":
        return min(logprobs)
    k = max(1, math.ceil(0.1 * len(logprobs)))
    return sum(sorted(logprobs)[:k]) / k


def softmax_of(scores):
    top = max(scores.values())
    weights = {key: math.exp(value - top) for key, value in scores.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def sigmoid(value):
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def load_cells(root, split, match=""):
    """One candidate-scored cell per model; two cells for a model is an error, never a silent choice."""
    found = {}
    for path in sorted(root.glob(f"*_cs*/{split}.jsonl")):
        if match and match not in path.parent.name:
            continue
        found.setdefault(base_model(path.parent.name), []).append(path)
    clashes = {model: paths for model, paths in found.items() if len(paths) > 1}
    if clashes:
        listing = "\n".join(f"  {model}:\n" + "\n".join(f"    {p.parent.name}" for p in paths)
                            for model, paths in clashes.items())
        raise SystemExit("several candidate-scored cells per model; pass --match to pick one "
                         f"(e.g. --match _full_ or --match _n100_):\n{listing}")
    cells = {}
    for model, (path,) in found.items():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        cells[model] = {row["id"]: row for row in rows}
    return cells


def majority_of(task, answers):
    candidates = []
    for answer in answers:
        if answer is not None and not any(task.equivalent(c, answer) for c in candidates):
            candidates.append(answer)
    if not candidates:
        return None
    counts = {c: sum(task.equivalent(a, c) for a in answers if a) for c in candidates}
    return max(candidates, key=lambda c: (counts[c], -candidates.index(c)))


def main():
    args = parse_args()
    task = get_task(args.task)
    cells = load_cells(Path(args.store, args.task), args.split, args.match)
    if not cells:
        print(f"no candidate-scored cells under {Path(args.store, args.task)} (need --score-candidates)")
        return
    shared = sorted(set.intersection(*(set(rows) for rows in cells.values())))
    counts = [len(((next(iter(cells.values()))[e].get("candidate_scores") or {}).get("candidates") or []))
              for e in shared]
    single = sum(c <= 1 for c in counts) / len(counts) if counts else 0
    print(f"{args.task}/{args.split}: {len(cells)} model(s), {len(shared)} shared example(s), "
          f"target={args.target}")
    print(f"candidate set: mean {sum(counts) / max(len(counts), 1):.2f} per example, "
          f"{100 * single:.0f}% have a single candidate"
          + ("   <- norm_* is degenerate on those; it can only encode disagreement" if single > 0.2 else ""))

    targets, correct = {}, {}
    for example_id in shared:
        rows = [cells[model][example_id] for model in cells]
        targets[example_id] = majority_of(task, [row["prediction"] for row in rows])
        gold = rows[0]["gold"]
        correct[example_id] = targets[example_id], gold

    print(f"\n{'model':<14}{'context':<10}{'estimator':<12}{'cov':>7}{'auroc':>8}{'auarc':>8}"
          f"{'ece':>8}{'sat>.99':>9}{'mean':>8}")
    for model, rows in sorted(cells.items()):
        # the incumbent baseline, on exactly these examples and this target
        support, labels = [], []
        for example_id in shared:
            row = rows[example_id]
            target = row["prediction"] if args.target == "own" else targets[example_id]
            samples = (row.get("sampled_answers") or {}).get("consistency_t0.7") or []
            valid = [a for a in samples if a is not None]
            if target is None or not valid:
                continue
            support.append(sum(task.equivalent(a, target) for a in valid) / len(valid))
            labels.append(float(task.is_correct(target, Example(example_id, "", row["gold"]))))
        if len(support) >= 10 and 0 < sum(labels) < len(labels):
            print(f"{model.replace('vllm__', '')[:13]:<14}{'samples':<10}{'consistency':<12}"
                  f"{len(support) / len(shared):>7.3f}{auroc(support, labels):>8.3f}"
                  f"{auarc(support, labels):>8.3f}{ece(support, labels):>8.3f}"
                  f"{sum(s > 0.99 for s in support) / len(support):>9.3f}"
                  f"{sum(support) / len(support):>8.3f}")
        for context in ("direct", "reasoned"):
            series = defaultdict(lambda: ([], []))
            for example_id in shared:
                row = rows[example_id]
                entry = row.get("candidate_scores") or {}
                target = row["prediction"] if args.target == "own" else targets[example_id]
                if target is None:
                    continue
                label = float(task.is_correct(target, Example(example_id, "", row["gold"])))
                scored = {c["text"]: c for c in entry.get("candidates", [])
                          if (c.get(context) or {}).get("logprobs")}
                chosen = next((c for c in scored.values() if task.equivalent(c["answer"], target)), None)
                if chosen is None:
                    continue
                for kind in RAW:
                    value = aggregate(chosen[context]["logprobs"], kind)
                    series[kind][0].append(value)
                    series[kind][1].append(label)
                for kind, base in (("norm_sum", "sum"), ("norm_mean", "mean")):
                    pool = {text: aggregate(c[context]["logprobs"], base) for text, c in scored.items()}
                    series[kind][0].append(softmax_of(pool)[chosen["text"]])
                    series[kind][1].append(label)
                nulls = [sum(lp) / len(lp) for lp in (chosen.get("null") or {}).values() if lp]
                if nulls:
                    own = aggregate(chosen[context]["logprobs"], "mean")
                    series["debiased"][0].append(sigmoid(own - max(nulls)))
                    series["debiased"][1].append(label)
            for kind in (*RAW, *PROBABILITY):
                scores, labels = series[kind]
                if len(scores) < 10 or not 0 < sum(labels) < len(labels):
                    continue
                saturated = sum(s > 0.99 for s in scores) / len(scores) if kind in PROBABILITY else float("nan")
                print(f"{model.replace('vllm__', '')[:13]:<14}{context:<10}{kind:<12}"
                      f"{len(scores) / len(shared):>7.3f}{auroc(scores, labels):>8.3f}"
                      f"{auarc(scores, labels):>8.3f}"
                      + (f"{ece(scores, labels):>8.3f}" if kind in PROBABILITY else f"{'—':>8}")
                      + (f"{saturated:>9.3f}" if kind in PROBABILITY else f"{'—':>9}")
                      + f"{sum(scores) / len(scores):>8.3f}")


if __name__ == "__main__":
    main()
