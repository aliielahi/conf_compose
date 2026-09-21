"""Experiment 01: which confidence estimator ranks a single model's own answers best, per task?"""

import argparse
import json
import time
from pathlib import Path

from conf_compose.constants import EVALUATION, RESULTS_DIR, TASKS
from conf_compose.data import get_task
from conf_compose.pipelines.inference import (STORE, InferenceSettings, by_model, describe, ensure_inference,
                                              load_model, load_records, missing)
from conf_compose.pipelines.report import confidence_report, format_report

NAME = "experiment01-zeroshot_estimators"
MODELS = ("vllm/q3-4bi", "vllm/l31-8bi", "vllm/g3-12i", "vllm/phi4mii")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=sorted(TASKS), choices=sorted(TASKS))
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--calibrator", choices=["beta", "platt"], default=EVALUATION["calibrator"])
    parser.add_argument("--n-boot", type=int, default=EVALUATION["n_boot"])
    parser.add_argument("--local-only", action="store_true", help="fail instead of loading a model")
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / NAME))
    return parser.parse_args()


def needed(args):
    return [InferenceSettings(task=task, model=model) for task in args.tasks for model in args.models]


def ensure_all(args):
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


def main():
    args = parse_args()
    ensure_all(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    for settings in needed(args):
        validation = load_records(settings, "validation", args.store)
        test = load_records(settings, "test", args.store)
        report = confidence_report(validation, test, args.calibrator, args.n_boot, answered_only=True)
        reports[f"{settings.task}/{settings.model}"] = report
        print(f"\n=== {settings.task} | {settings.model} ===")
        print("\n".join(format_report(report)))
    (out_dir / "reports.json").write_text(json.dumps(reports, indent=2))
    print(f"\nsaved {out_dir / 'reports.json'}")


if __name__ == "__main__":
    main()
