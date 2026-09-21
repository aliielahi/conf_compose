"""Fill the shared inference store: generate only the (task, model, decoding) cells that are missing."""

import argparse
import time

from conf_compose.constants import TASKS
from conf_compose.data import get_task
from conf_compose.pipelines.inference import (STORE, InferenceSettings, by_model, describe, ensure_inference,
                                              load_model, missing, split_count)
from conf_compose.pipelines.progress import Status


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True, choices=sorted(TASKS))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--voters", type=int, default=1, help="independent sampled answers per model")
    parser.add_argument("--answer-temperature", type=float, default=0.0, help="0 keeps greedy answers")
    parser.add_argument("--no-verbalized", action="store_true")
    parser.add_argument("--n-val", type=split_count, help="calibration examples: a count, `all`, or `none`")
    parser.add_argument("--n-test", type=split_count, help="evaluated examples: a count, `all`, or `none`")
    parser.add_argument("--all-signals", action="store_true",
                        help="also in-context verification and the content-free debiased sequence score")
    parser.add_argument("--dry-run", action="store_true", help="list what is missing and exit")
    parser.add_argument("--store", default=str(STORE))
    return parser.parse_args()


def requested(args):
    return [InferenceSettings(task=task, model=model, n_val=args.n_val, n_test=args.n_test,
                              answer_temperature=args.answer_temperature, voter=voter,
                              verbalized=not args.no_verbalized,
                              verification_context=args.all_signals, debias=args.all_signals)
            for task in args.tasks for model in args.models
            for voter in range(args.voters if args.answer_temperature > 0 else 1)]


def main():
    args = parse_args()
    wanted = requested(args)
    pending = missing(wanted, args.store)
    print(f"store {args.store}: {len(wanted)} requested, {len(pending)} missing")
    if pending:
        print(describe(pending))
    if args.dry_run or not pending:
        return

    grouped = by_model(pending)
    status = Status(f"inference {'+'.join(args.tasks)}", total=len(pending))
    for index, (model, group) in enumerate(grouped.items(), 1):
        status.stage(f"loading {model} ({index}/{len(grouped)} models)")
        start = time.time()
        llm = load_model(model)
        status.stage(f"{model} ready in {time.time() - start:.0f}s, {len(group)} cell(s)")
        for settings in group:
            began = time.time()
            path = ensure_inference(settings, llm, get_task(settings.task), args.store)
            status.step(f"{path.name} | {time.time() - began:.0f}s")
    status.finish(f"all {len(wanted)} cell(s) present, {len(missing(wanted, args.store))} missing")


if __name__ == "__main__":
    main()
