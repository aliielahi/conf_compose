"""Score one shared target per example with every model's verifier: the anchor answer and the majority vote."""

import argparse
import glob
import json
from pathlib import Path

from conf_compose.confidence_estimators import SelfVerification, Target
from conf_compose.constants import CACHE_DIR, RESULTS_DIR, TASKS, VLLM
from conf_compose.data import Example, get_task
from conf_compose.utils.llm_calls import LLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--model", required=True, help="the verifier, as a provider/model spec")
    parser.add_argument("--root", required=True, help="the sweep's task directory holding every voter run")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--targets", nargs="+", default=["anchor", "majority"], choices=["anchor", "majority"])
    parser.add_argument("--out", help="default: alongside the verifier's own records")
    parser.add_argument("--gpu-memory-utilization", type=float, default=VLLM["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=VLLM["max_model_len"])
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    return parser.parse_args()


def shared_examples(task, paths):
    """Every voter's prediction per example id, in voter order, over the ids all voters answered."""
    per_voter = [{row["id"]: row for row in _read(path)} for path in paths]
    ids = sorted(set.intersection(*(set(rows) for rows in per_voter)), key=_sort_key)
    return [(example_id, [rows[example_id] for rows in per_voter]) for example_id in ids]


def targets_for(task, mode, rows_by_example):
    """One target answer per example, either the first voter's or the ordinary majority vote."""
    out = {}
    for example_id, rows in rows_by_example:
        answers = [row["prediction"] for row in rows if row["prediction"] is not None]
        if not answers:
            continue
        out[example_id] = answers[0] if mode == "anchor" else _majority(task, answers)
    return out


def main():
    args = parse_args()
    task = get_task(args.task)
    engine = {"max_num_seqs": VLLM["max_num_seqs"], "max_num_batched_tokens": VLLM["max_num_batched_tokens"]}
    llm = LLM(args.model, cache_dir=args.cache_dir, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, quiet=True, engine_kwargs=engine)
    verifier = SelfVerification(llm, claim=True)

    for split in args.splits:
        paths = sorted(_expand([f"{args.root}/*/{split}.jsonl"]))
        if not paths:
            print(f"{split}: no records under {args.root}")
            continue
        rows_by_example = shared_examples(task, paths)
        print(f"{args.task}/{split}: {len(paths)} voters, {len(rows_by_example)} shared examples")
        first = {example_id: rows[0] for example_id, rows in rows_by_example}

        scores = {}
        for mode in args.targets:
            answers = targets_for(task, mode, rows_by_example)
            ids = [example_id for example_id, _ in rows_by_example if example_id in answers]
            targets = [Target(example=Example(example_id, first[example_id].get("question", ""),
                                              first[example_id]["gold"]),
                              conversation=[], response="", answer=answers[example_id]) for example_id in ids]
            values = verifier.estimate(task, targets)
            print(f"  {mode}: scored {sum(v is not None for v in values)}/{len(ids)} "
                  f"(missing labels {verifier.missing_labels})")
            for example_id, value in zip(ids, values):
                scores.setdefault(example_id, {})[mode] = {"answer": answers[example_id], "verification": value}

        out = Path(args.out or args.root) / f"target_verification_{_tag(args.model)}_{split}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": args.model, "task": args.task, "split": split,
                                   "voters": [Path(p).parent.name for p in paths], "scores": scores}, indent=2))
        print(f"  saved {out}")


def _tag(model):
    return model.replace("/", "__")


def _majority(task, answers):
    candidates = []
    for answer in answers:
        if not any(task.equivalent(existing, answer) for existing in candidates):
            candidates.append(answer)
    counts = {c: sum(task.equivalent(a, c) for a in answers) for c in candidates}
    return max(candidates, key=lambda c: (counts[c], -candidates.index(c)))


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _expand(paths):
    expanded = []
    for path in paths:
        expanded += sorted(glob.glob(path)) or [path]
    return expanded


def _sort_key(example_id):
    tail = example_id.rsplit("-", 1)[-1]
    return (int(tail), example_id) if tail.isdigit() else (0, example_id)


if __name__ == "__main__":
    main()
