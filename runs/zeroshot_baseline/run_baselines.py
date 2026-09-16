"""Run eval_zero_shot for every model of a sweep (all tasks per model load), optionally several models in parallel."""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from conf_compose.constants import BASELINES, LOGS_DIR, RESULTS_DIR, ROOT, TASKS, VLLM
from conf_compose.pipelines.runs import run_dir


def pending_tasks(results: Path, model: str, tasks, force: bool):
    if force:
        return list(tasks)
    done = {task for task in tasks
            if (run_dir(str(results), task, model, TASKS[task]["n_val"], TASKS[task]["n_test"]) / "report.json").exists()}
    return [task for task in tasks if task not in done]


def run_model(model: str, tasks, results: Path, logs: Path, parallel: int, extra) -> tuple:
    log = logs / f"{model.replace('/', '__')}.log"
    memory = VLLM["gpu_memory_utilization"] / parallel
    command = [sys.executable, str(Path(__file__).parent / "eval_zero_shot.py"), "--model", model, "--tasks", *tasks,
               "--out-dir", str(results), "--gpu-memory-utilization", f"{memory:.3f}", "--skip-existing", *extra]
    print(f"[{time.strftime('%H:%M:%S')}] start {model} | {' '.join(tasks)} | {log}", flush=True)
    start = time.time()
    with log.open("w") as handle:
        code = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
    result = "ok" if code == 0 else f"failed ({code})"
    minutes = (time.time() - start) / 60
    print(f"[{time.strftime('%H:%M:%S')}] {result} {model} | {minutes:.1f} min", flush=True)
    return model, result, minutes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True, help="results go to results/<sweep>, logs to logs/<sweep>")
    parser.add_argument("--models", nargs="+", default=BASELINES["models"])
    parser.add_argument("--tasks", nargs="+", default=BASELINES["tasks"])
    parser.add_argument("--parallel", type=int, default=1, help="models sharing the GPU at once")
    parser.add_argument("--force", action="store_true", help="rerun even if report.json exists")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="arguments after -- go to eval_zero_shot.py")
    args = parser.parse_args()
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra

    results, logs = RESULTS_DIR / args.sweep, LOGS_DIR / args.sweep
    logs.mkdir(parents=True, exist_ok=True)
    jobs = [(model, pending_tasks(results, model, args.tasks, args.force)) for model in args.models]
    for model, tasks in jobs:
        if not tasks:
            print(f"skip {model}: all tasks done", flush=True)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run_model, model, tasks, results, logs, args.parallel, extra)
                   for model, tasks in jobs if tasks]
        status = [future.result() for future in futures]

    print("\n=== summary ===")
    for model, result, minutes in status:
        print(f"{model:<20} {result:<12} {minutes:6.1f} min")


if __name__ == "__main__":
    main()
