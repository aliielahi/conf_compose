"""Score one shared target per example with every model's verifier: the anchor answer and the majority vote."""

import argparse
import json
from pathlib import Path

from conf_compose.composition import shared_targets, voter_rows
from conf_compose.confidence_estimators import SelfVerification, Target
from conf_compose.constants import CACHE_DIR, TASKS, VLLM
from conf_compose.data import Example, get_task
from conf_compose.pipelines.runs import expand_globs
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


def main():
    args = parse_args()
    task = get_task(args.task)
    engine = {"max_num_seqs": VLLM["max_num_seqs"], "max_num_batched_tokens": VLLM["max_num_batched_tokens"]}
    llm = LLM(args.model, cache_dir=args.cache_dir, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, quiet=True, engine_kwargs=engine)
    verifier = SelfVerification(llm, claim=True)

    for split in args.splits:
        paths = sorted(expand_globs([f"{args.root}/*/{split}.jsonl"]))
        if not paths:
            print(f"{split}: no records under {args.root}")
            continue
        rows_by_example = voter_rows(paths)
        print(f"{args.task}/{split}: {len(paths)} voters, {len(rows_by_example)} shared examples")
        first = {example_id: rows[0] for example_id, rows in rows_by_example}

        scores = {}
        for mode in args.targets:
            answers = shared_targets(task, mode, rows_by_example)
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


if __name__ == "__main__":
    main()
