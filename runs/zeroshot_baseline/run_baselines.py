"""Run the zero-shot confidence baseline grid from constants.json: one eval process per model, all its tasks."""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from conf_compose.constants import BASELINES, LOGS_DIR, RESULTS_DIR, ROOT, TASKS, VLLM
from conf_compose.pipelines.runs import run_dir


def jobs_from_grid(grid, results: Path, force: bool):
    """One (model, tasks) job per model in the grid, keeping only tasks without a report."""
    per_model: dict = {}
    for group in grid:
        for model in group["models"]:
            for task in group["tasks"]:
                done = (run_dir(str(results), task, model, TASKS[task]["n_val"],
                                TASKS[task]["n_test"]) / "report.json").exists()
                if force or not done:
                    per_model.setdefault(model, [])
                    if task not in per_model[model]:
                        per_model[model].append(task)
    return list(per_model.items())


def run_model(model: str, tasks, results: Path, logs: Path, parallel: int, extra) -> tuple:
    log = logs / f"{model.replace('/', '__')}.log"
    memory = VLLM["gpu_memory_utilization"] / parallel
    command = [sys.executable, str(Path(__file__).parent / "eval_zero_shot.py"), "--model", model, "--tasks", *tasks,
               "--out-dir", str(results), "--gpu-memory-utilization", f"{memory:.3f}", "--skip-existing", *extra]
    print(f"[{time.strftime('%H:%M:%S')}] start {model} | {' '.join(tasks)} | {log}", flush=True)
    start = time.time()
    with log.open("w") as handle:
        code = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
    minutes = (time.time() - start) / 60
    result = "ok" if code == 0 else f"failed ({code})"
    print(f"[{time.strftime('%H:%M:%S')}] {result} {model} | {minutes:.1f} min", flush=True)
    return model, result, minutes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default=BASELINES["sweep"], help="results/<sweep> and logs/<sweep>")
    parser.add_argument("--models", nargs="+", help="override the grid's models")
    parser.add_argument("--tasks", nargs="+", help="override the grid's tasks")
    parser.add_argument("--parallel", type=int, default=1, help="models sharing the GPU at once")
    parser.add_argument("--force", action="store_true", help="rerun even if report.json exists")
    parser.add_argument("--dry-run", action="store_true", help="print the pending jobs and exit")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="arguments after -- go to eval_zero_shot.py")
    args = parser.parse_args()
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra

    grid = BASELINES["grid"]
    if args.models or args.tasks:
        models = args.models or sorted({m for group in grid for m in group["models"]})
        tasks = args.tasks or sorted({t for group in grid for t in group["tasks"]})
        grid = [{"models": models, "tasks": tasks}]

    results, logs = RESULTS_DIR / args.sweep, LOGS_DIR / args.sweep
    jobs = jobs_from_grid(grid, results, args.force)
    print(f"sweep {args.sweep}: {len(jobs)} model(s) pending", flush=True)
    for model, tasks in jobs:
        print(f"  {model}: {' '.join(tasks)}", flush=True)
    if args.dry_run or not jobs:
        return

    logs.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run_model, model, tasks, results, logs, args.parallel, extra) for model, tasks in jobs]
        status = [future.result() for future in futures]

    print("\n=== summary ===")
    for model, result, minutes in status:
        print(f"{model:<20} {result:<12} {minutes:6.1f} min")


if __name__ == "__main__":
    main()
