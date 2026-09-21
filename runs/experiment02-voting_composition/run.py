"""Experiment 02: does pooling confidence across independent voters beat one voter, at matched budget?"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from conf_compose.composition import (FAMILIES, GridConfig, attach_target_scores, best_by_validation,
                                      build_panels, collect, evaluate, fit_intercepts, format_cell,
                                      format_table, from_zero_shot, rows_from)
from conf_compose.constants import RESULTS_DIR
from conf_compose.data import get_task
from conf_compose.pipelines.inference import (STORE, by_model, describe, ensure_inference, inference_dir,
                                              load_model, missing, split_count, voter_settings)
from conf_compose.pipelines.progress import Status
from conf_compose.pipelines.runs import source_hash

NAME = "experiment02-voting_composition"
TASKS = ("csqa", "boolq")
MODELS = ("vllm/q3-4bi", "vllm/l31-8bi", "vllm/g3-12i", "vllm/phi4mii")
VOTERS = 5
TARGETS = ("anchor", "majority")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--voters", type=int, default=VOTERS)
    parser.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--samples", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--cap", type=int, default=5)
    parser.add_argument("--n-val", type=split_count, help="must match the inference sweep: a count, all, none")
    parser.add_argument("--n-test", type=split_count, help="must match the inference sweep")
    parser.add_argument("--all-signals", action="store_true", help="must match the inference sweep")
    parser.add_argument("--no-verbalized", action="store_true", help="must match the inference sweep")
    parser.add_argument("--holdout", type=float, default=0.3,
                        help="fraction of test used for selection when no validation split was generated")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--local-only", action="store_true", help="fail instead of loading a model")
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / NAME))
    return parser.parse_args()


def needed(args):
    """Exactly the settings the inference sweep wrote, so an existing store is never regenerated."""
    return [s for task in args.tasks
            for s in voter_settings(task, args.models, args.voters, verbalized=not args.no_verbalized,
                                    n_val=args.n_val, n_test=args.n_test,
                                    verification_context=args.all_signals, debias=args.all_signals)]


def ensure_all(args):
    """The store is the contract: generate what is missing, or say what a local run still owes."""
    pending = missing(needed(args), args.store)
    if not pending:
        print(f"inference store complete ({len(needed(args))} cells)")
        return
    if args.local_only:
        raise SystemExit(f"{len(pending)} inference(s) missing and --local-only was set:\n{describe(pending)}")
    for model, group in by_model(pending).items():
        llm = load_model(model)
        print(f"loaded {llm}", flush=True)
        for settings in group:
            began = time.time()
            print(f"  {ensure_inference(settings, llm, get_task(settings.task), args.store).name}"
                  f" | {time.time() - began:.0f}s", flush=True)


def split_paths(task, args, split):
    paths = {inference_dir(s, args.store).name: inference_dir(s, args.store) / f"{split}.jsonl"
             for s in needed_for(task, args)}
    return {name: path for name, path in paths.items() if path.exists()}


def needed_for(task, args):
    return [s for s in needed(args) if s.task == task]


def load_splits(task_name, args):
    """Validation and test items; when the sweep generated no validation, hold out part of test instead."""
    test = from_zero_shot(split_paths(task_name, args, "test"))
    validation_paths = split_paths(task_name, args, "validation")
    if validation_paths:
        return from_zero_shot(validation_paths), test, False
    cut = int(len(test) * args.holdout)
    ordered = sorted(test, key=lambda item: hashlib.sha256(item.example_id.encode()).hexdigest())
    return ordered[:cut], ordered[cut:], True


def run_cell(task_name, target, args):
    task = get_task(task_name)
    config = GridConfig(target, args.sizes, args.samples, args.families, "add_half", args.cap)
    validation, test, carved = load_splits(task_name, args)
    if carved:
        print(f"  no validation split generated: holding out {len(validation)} of "
              f"{len(validation) + len(test)} test examples for selection and calibration")
    scores = sorted(Path(args.store, task_name).glob("target_verification_*.json"))
    if scores:
        print(f"  target scores: {attach_target_scores(validation, [str(p) for p in scores if 'validation' in p.name], target)}"
              f" validation, {attach_target_scores(test, [str(p) for p in scores if 'test' in p.name], target)} test")

    models = [s.model for s in sorted(validation[0].round_streams((0,)), key=lambda s: s.agent)]
    panels = build_panels(models, config)
    print(f"  {task_name}/{target}: validation={len(validation)} test={len(test)} panels={len(panels)}", flush=True)

    val_rows = rows_from(*collect(task, validation, panels, config, progress=100), total=len(validation))
    val_report = evaluate(val_rows, reference="", n_boot=0)
    reference = best_by_validation(val_rows, val_report, "single|")
    test_rows = rows_from(*collect(task, test, panels, config, progress=100), total=len(test))
    report = evaluate(test_rows, reference, args.n_boot, fit_intercepts(val_rows))

    out_dir = Path(args.out_dir) / task_name / f"panels_{target}_{_cell_id(args, task_name)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"reference": reference, "report": report, "validation": val_report}, indent=2))
    (out_dir / "predictions.json").write_text(json.dumps(
        {r.method: {"ids": r.example_ids, "scores": r.scores, "correct": r.correct} for r in test_rows}))
    (out_dir / "manifest.json").write_text(json.dumps(
        {"experiment": NAME, "args": vars(args), "task": task_name, "target": target, "models": models,
         "inferences": sorted(split_paths(task_name, args, "test")), "holdout_from_test": carved,
         "code": source_hash("conf_compose.composition")}, indent=2))
    print(f"  saved {out_dir}", flush=True)
    return report, val_report


def main():
    args = parse_args()
    status = Status(NAME, total=len(args.tasks) * len(args.targets))
    status.stage("checking the inference store")
    ensure_all(args)
    cells = {}
    for task_name in args.tasks:
        for target in args.targets:
            status.stage(f"{task_name} / {target}")
            cells[(task_name, target)] = run_cell(task_name, target, args)
            status.step(f"{task_name}/{target}")

    summary = []
    for (task_name, target), (report, val_report) in sorted(cells.items()):
        summary.append(f"\n{'=' * 78}\n{task_name} / target={target} / rule=mean / metric=auroc")
        summary += format_cell(report, val_report, "auroc", "mean")
    text = "\n".join(summary)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "summary.txt").write_text(text)
    print(text)
    status.finish(f"saved {Path(args.out_dir) / 'summary.txt'}")


def _cell_id(args, task_name):
    payload = json.dumps({"task": task_name, "models": args.models, "voters": args.voters,
                          "sizes": args.sizes, "samples": args.samples, "families": args.families,
                          "cap": args.cap, "code": source_hash("conf_compose.composition")}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


if __name__ == "__main__":
    main()
