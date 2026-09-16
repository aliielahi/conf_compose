"""Run eval_zero_shot for every model x task of a sweep, one process per run, skipping finished runs."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from conf_compose.constants import BASELINES, LOGS_DIR, RESULTS_DIR, ROOT, TASKS
from conf_compose.pipelines.runs import run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True, help="results go to results/<sweep>, logs to logs/<sweep>")
    parser.add_argument("--models", nargs="+", default=BASELINES["models"])
    parser.add_argument("--tasks", nargs="+", default=BASELINES["tasks"])
    parser.add_argument("--force", action="store_true", help="rerun even if report.json exists")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="arguments after -- go to eval_zero_shot.py")
    args = parser.parse_args()
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra

    results, logs = RESULTS_DIR / args.sweep, LOGS_DIR / args.sweep
    logs.mkdir(parents=True, exist_ok=True)
    status = []
    for task in args.tasks:
        for model in args.models:
            output = run_dir(str(results), task, model, TASKS[task]["n_val"], TASKS[task]["n_test"])
            if (output / "report.json").exists() and not args.force:
                status.append((task, model, "skipped", 0.0))
                continue
            log = logs / f"{task}_{model.replace('/', '__')}.log"
            command = [sys.executable, str(Path(__file__).parent / "eval_zero_shot.py"), "--model", model,
                       "--task", task, "--out-dir", str(results), *extra]
            print(f"[{time.strftime('%H:%M:%S')}] start {task} | {model} | {log}", flush=True)
            start = time.time()
            with log.open("w") as handle:
                code = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
            result = "ok" if code == 0 else f"failed ({code})"
            status.append((task, model, result, time.time() - start))
            print(f"[{time.strftime('%H:%M:%S')}] {result} {task} | {model} | {status[-1][3] / 60:.1f} min", flush=True)

    print("\n=== summary ===")
    for task, model, result, seconds in status:
        print(f"{task:<10} {model:<20} {result:<12} {seconds / 60:6.1f} min")


if __name__ == "__main__":
    main()
