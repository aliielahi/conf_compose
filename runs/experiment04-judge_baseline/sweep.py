"""Sweep the judge over tasks, panels and judges, showing one confidence estimator at a time."""

import argparse
import csv
import importlib.util
import time
import traceback
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from conf_compose.constants import RESULTS_DIR
from conf_compose.pipelines.inference import STORE, load_model

NAME = "experiment04-judge_baseline"
_spec = importlib.util.spec_from_file_location("jrun", Path(__file__).with_name("run.py"))
jrun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jrun)

PANELS = [
    ["q3-8bi", "l31-8bi"],
    ["g2-9i", "phi4mii"],
    ["q3-4bi", "q3-8bi"],
    ["g2-9i", "g3-12i"],
    ["q3-8bi", "g2-9i", "phi4mii"],
    ["q3-4bi", "q3-8bi", "l31-8bi"],
    ["q3-4bi", "l31-8bi", "g3-12i"],
    ["g2-9i", "g3-12i", "phi4mii"],
    ["q3-8bi", "l31-8bi", "g2-9i", "phi4mii"],
    ["q3-4bi", "q3-8bi", "l31-8bi", "g2-9i"],
    ["q3-8bi", "l31-8bi", "g3-12i", "phi4mii"],
    ["q3-4bi", "q3-8bi", "l31-8bi", "g2-9i", "phi4mii"],
    ["q3-4bi", "q3-8bi", "l32-3bi", "l31-8bi", "g2-9i"],
    ["q3-4bi", "q3-8bi", "l31-8bi", "g2-9i", "g3-12i", "phi4mii"],
    ["q3-4bi", "q3-8bi", "l32-3bi", "l31-8bi", "g2-9i", "phi4mii"],
]
JUDGES = ["vllm/g3-27i", "vllm/l32-3bi"]
# Cheapest split first, so an interrupted night still leaves whole datasets finished.
TASKS = ["gpqa", "truthfulqa", "csqa", "gsm8k", "boolq"]
METHODS = ["consistency_t0.7", "cand_direct_sum"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judges", nargs="+", default=JUDGES)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        help="each is a separate run; the judge never sees two at once")
    parser.add_argument("--panel-sizes", nargs="+", type=int, help="only panels of these sizes")
    parser.add_argument("--no-control", action="store_true",
                        help="skip the reasoning-only arm the confidence arms are compared against")
    parser.add_argument("--modes", nargs="+", default=["verbalized"])
    parser.add_argument("--voter", type=int, default=0)
    parser.add_argument("--match", default="_cs7s")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, help="first N examples; omit for the whole split")
    parser.add_argument("--word-limit", type=int, default=150)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--log", default="logs/current_run.log",
                        help="one running account of the whole sweep, appended to across launches")
    parser.add_argument("--force", action="store_true", help="redo runs already on disk")
    parser.add_argument("--dry-run", action="store_true", help="list the plan and exit")
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / NAME))
    return parser.parse_args()


def plan(args):
    """One entry per saved cell: judge, task, panel, and the single signal shown."""
    panels = [p for p in PANELS if not args.panel_sizes or len(p) in args.panel_sizes]
    arms = ([] if args.no_control else [("reasoning", args.methods[0])]) \
        + [("reasoning_confidence", method) for method in args.methods]
    return [(judge, task, panel, view, method)
            for judge in args.judges for task in args.tasks
            for panel in panels for view, method in arms]


def case(args, judge, task, panel, view, method):
    return Namespace(**{**vars(args), "judge": judge, "task": task,
                        "panel": [f"vllm/{model}" for model in panel],
                        "view": view, "confidence_method": method})


def log(path, message):
    """One timestamped line, closed each time, so a kill -9 cannot lose what already happened."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}\n")


def summarise(path, args, pending, rows, failures, started):
    """Everything this launch did, appended whether it finished, crashed or was interrupted."""
    elapsed = time.time() - started
    lines = [f"SUMMARY  {len(rows)} ok, {len(failures)} failed, {len(pending) - len(rows) - len(failures)} "
             f"not reached, of {len(pending)} planned in {elapsed / 3600:.2f}h"]
    for task in args.tasks:
        done = [r for r in rows if r["task"] == task]
        if done:
            lines.append(f"  {task:<12}{len(done):>4} run(s)   "
                         f"mean majority_acc {sum(r['majority_acc'] for r in done) / len(done):.3f}")
    for shown in ["none"] + list(args.methods):
        arm = [r for r in rows if r["shown_confidence"] == shown and r.get("verbalized_auroc")]
        if arm:
            lines.append(f"  shown={shown:<22}{len(arm):>4} run(s)   "
                         f"mean auroc {sum(r['verbalized_auroc'] for r in arm) / len(arm):.3f}   "
                         f"mean ece {sum(r['verbalized_ece'] for r in arm) / len(arm):.3f}")
    for label, error in failures:
        lines.append(f"  FAILED  {label}: {error}")
    for line in lines:
        print(line)
    log(path, "\n".join(lines) + f"\n{'-' * 78}")


def main():
    args = parse_args()
    work = plan(args)
    out, logfile = Path(args.out_dir), Path(args.log)
    pending = [c for c in work
               if args.force or not (out / c[1] / jrun._cell(case(args, *c)) / "verdicts.jsonl").exists()]
    header = (f"{len(work)} run(s) planned, {len(work) - len(pending)} already on disk, "
              f"{len(pending)} to go")
    print(header)
    print(f"judges={len(args.judges)} tasks={len(args.tasks)} panels={len(set(map(str, [c[2] for c in work])))} "
          f"arms={'' if args.no_control else 'control+'}{len(args.methods)} method(s)")
    if args.dry_run or not pending:
        for judge, task, panel, view, method in pending:
            print(f"  {judge:<14}{task:<12}{'+'.join(panel):<46}{view:<22}{method}")
        return

    log(logfile, f"{'=' * 78}\nLAUNCH   judges={','.join(args.judges)} tasks={','.join(args.tasks)} "
                 f"methods={','.join(args.methods)} limit={args.limit or 'full'}\n         {header}")
    out.mkdir(parents=True, exist_ok=True)
    rows, failures, done, started = [], [], 0, time.time()
    try:
        for judge in args.judges:
            batch = [c for c in pending if c[0] == judge]
            if not batch:
                continue
            print(f"\n{'#' * 78}\n# loading judge {judge} for {len(batch)} run(s)\n{'#' * 78}")
            log(logfile, f"JUDGE    {judge}: loading for {len(batch)} run(s)")
            try:
                llm = load_model(judge)
            except Exception as error:
                log(logfile, f"FATAL    {judge} failed to load: {type(error).__name__}: {error}")
                failures.append((judge, f"load failed: {error}"))
                traceback.print_exc()
                continue
            for judge_name, task, panel, view, method in batch:
                done += 1
                label = f"{judge_name} {task} {'+'.join(panel)} {view} {method}"
                print(f"\n{'=' * 78}\n[{done}/{len(pending)}] {label}")
                log(logfile, f"START    [{done}/{len(pending)}] {label}")
                clock = time.time()
                try:
                    row = jrun.judge_once(case(args, judge_name, task, panel, view, method), llm)
                    rows.append(row)
                    log(logfile, f"OK       [{done}/{len(pending)}] n={row['n']} "
                                 f"acc={row['majority_acc']:.3f} "
                                 f"auroc={row.get('verbalized_auroc', float('nan')):.3f} "
                                 f"ece={row.get('verbalized_ece', float('nan')):.3f} "
                                 f"{time.time() - clock:.0f}s")
                except SystemExit as error:
                    print(f"SKIPPED: {error}")
                    failures.append((label, f"skipped: {error}"))
                    log(logfile, f"SKIP     [{done}/{len(pending)}] {error}")
                except Exception as error:
                    print(f"CRASHED: {type(error).__name__}: {error}")
                    traceback.print_exc()
                    failures.append((label, f"{type(error).__name__}: {error}"))
                    log(logfile, f"CRASH    [{done}/{len(pending)}] {type(error).__name__}: {error}\n"
                                 + traceback.format_exc().rstrip())
                write(out / "summary.csv", rows)
            del llm
    except KeyboardInterrupt:
        log(logfile, "STOPPED  interrupted by the user")
        print("\ninterrupted")
    finally:
        summarise(logfile, args, pending, rows, failures, started)
        print(f"summary.csv -> {out / 'summary.csv'}   log -> {logfile}")


def write(path, rows):
    """Rewritten after every run, so an interrupted night still leaves the finished rows behind."""
    if not rows:
        return
    fields = list({key: None for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
