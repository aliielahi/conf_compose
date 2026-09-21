"""Every model verifies the panel's shared target answer, so the majority vote is rated by a real verifier."""

import argparse
import json
from pathlib import Path

from conf_compose.composition import shared_targets, voter_rows
from conf_compose.confidence_estimators import SelfVerification, Target
from conf_compose.constants import TASKS
from conf_compose.data import Example, get_task
from conf_compose.pipelines.inference import (STORE, inference_dir, load_model, split_count,
                                              voter_settings)

MODELS = ("vllm/q3-4bi", "vllm/l31-8bi", "vllm/g3-12i", "vllm/phi4mii")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), help="the panel, and the verifiers")
    parser.add_argument("--voters", type=int, default=5)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--targets", nargs="+", default=["anchor", "majority"], choices=["anchor", "majority"])
    parser.add_argument("--n-val", type=split_count, help="must match the inference sweep")
    parser.add_argument("--n-test", type=split_count, help="must match the inference sweep")
    parser.add_argument("--all-signals", action="store_true", help="must match the inference sweep")
    parser.add_argument("--no-verbalized", action="store_true", help="must match the inference sweep")
    parser.add_argument("--store", default=str(STORE))
    return parser.parse_args()


def panel_paths(args, split):
    return [str(inference_dir(s, args.store) / f"{split}.jsonl")
            for s in voter_settings(args.task, args.models, args.voters, verbalized=not args.no_verbalized,
                                    n_val=args.n_val, n_test=args.n_test,
                                    verification_context=args.all_signals, debias=args.all_signals)]


def main():
    args = parse_args()
    task = get_task(args.task)
    out_root = Path(args.store) / args.task
    for model in args.models:
        llm = load_model(model)
        verifier = SelfVerification(llm, claim=True)
        print(f"\n=== {model} ===", flush=True)
        for split in args.splits:
            paths = [p for p in panel_paths(args, split) if Path(p).exists()]
            if not paths:
                print(f"  {split}: no panel records under {out_root}")
                continue
            rows_by_example = voter_rows(paths)
            first = {example_id: rows[0] for example_id, rows in rows_by_example}
            scores = {}
            for mode in args.targets:
                answers = shared_targets(task, mode, rows_by_example)
                ids = [i for i, _ in rows_by_example if i in answers]
                targets = [Target(example=Example(i, first[i].get("question", ""), first[i]["gold"]),
                                  conversation=[], response="", answer=answers[i]) for i in ids]
                values = verifier.estimate(task, targets)
                print(f"  {split}/{mode}: scored {sum(v is not None for v in values)}/{len(ids)} "
                      f"(missing labels {verifier.missing_labels})", flush=True)
                for example_id, value in zip(ids, values):
                    scores.setdefault(example_id, {})[mode] = {"answer": answers[example_id],
                                                               "verification": value}
            out = out_root / f"target_verification_{model.replace('/', '__')}_{split}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"model": model, "task": args.task, "split": split,
                                       "voters": [Path(p).parent.name for p in paths], "scores": scores},
                                      indent=2))
            print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
