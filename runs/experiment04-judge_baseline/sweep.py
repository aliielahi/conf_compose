"""Sweep the judge over tasks, panels and judges, showing one confidence estimator at a time."""

import argparse
import csv
import importlib.util
import json
import time
import traceback
from argparse import Namespace
from datetime import datetime, timedelta
from pathlib import Path

from conf_compose.constants import COMPOSITION_PANELS, RESULTS_DIR
from conf_compose.pipelines.inference import STORE

NAME = "experiment04-judge_baseline"
LOG_DIR = Path("logs") / NAME
_spec = importlib.util.spec_from_file_location("jrun", Path(__file__).with_name("run.py"))
jrun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jrun)

PANELS = COMPOSITION_PANELS
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
    parser.add_argument("--log", default=str(LOG_DIR / "current_run.log"),
                        help="one running account of the sweep, appended to across launches")
    parser.add_argument("--force", action="store_true", help="redo cells that are already complete")
    parser.add_argument("--dry-run", action="store_true", help="list the plan and exit")
    parser.add_argument("--collect", action="store_true", help="only rebuild summary.csv from the cells on disk")
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


def complete(cell):
    """Done only if scored under the current context window; anything older is redone."""
    path = cell / "metrics.json"
    return path.exists() and json.loads(path.read_text()).get("context") == jrun.JUDGE_CONTEXT


def split_size(args, task):
    """Examples one run of this task will judge, read off a panel cell without loading it."""
    path = next(p for p in sorted(Path(args.store, task).glob(f"*/{args.split}.jsonl"))
                if args.match in p.parent.name and f"v{args.voter}_" in p.parent.name)
    size = sum(1 for line in path.open() if line.strip())
    return min(size, args.limit) if args.limit else size


def collect(out):
    """summary.csv from every cell on disk, so resumes and separate judge processes never overwrite each other."""
    rows = []
    for cell in sorted(out.glob("*/*/verdicts.jsonl")):
        cell = cell.parent
        path = cell / "metrics.json"
        if not path.exists():
            # Cells from before metrics.json existed: scored once from their verdicts, then cached.
            manifest = json.loads((cell / "manifest.json").read_text())
            saved = Namespace(**manifest["args"])
            verdicts = [json.loads(line) for line in (cell / "verdicts.jsonl").open() if line.strip()]
            summary = {**jrun.describe(saved), **jrun.metrics(verdicts, saved.modes)}
            summary["context"] = manifest.get("context", 8192)
            path.write_text(json.dumps(summary, indent=2))
        rows.append(json.loads(path.read_text()))
    if rows:
        fields = list({key: None for row in rows for key in row})
        with (out / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def log(path, message):
    """One timestamped line, closed each time, so a kill -9 cannot lose what already happened."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}\n")


def clock(seconds):
    return str(timedelta(seconds=int(seconds)))


def summarise(path, args, pending, rows, failures, started):
    """Everything this launch did, appended whether it finished, crashed or was interrupted."""
    lines = [f"SUMMARY  {len(rows)} ok, {len(failures)} failed, {len(pending) - len(rows) - len(failures)} "
             f"not reached, of {len(pending)} planned in {clock(time.time() - started)}"]
    for task in args.tasks:
        done = [r for r in rows if r["task"] == task]
        if done:
            truncated = sum(r.get("truncated") or 0 for r in done)
            lines.append(f"  {task:<12}{len(done):>4} run(s)   mean majority_acc "
                         f"{sum(r['majority_acc'] for r in done) / len(done):.3f}   truncated outputs {truncated}   "
                         f"longest prompt {max(r.get('max_prompt_tokens') or 0 for r in done)}")
    for shown in ["none"] + list(args.methods):
        arm = [r for r in rows if r["shown_confidence"] == shown and r.get("verbalized_auroc") is not None]
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
    out, logfile = Path(args.out_dir), Path(args.log)
    if args.collect:
        rows = collect(out)
        print(f"{len(rows)} cell(s) -> {out / 'summary.csv'}")
        return

    work = plan(args)
    pending = [c for c in work if args.force or not complete(out / c[1] / jrun._cell(case(args, *c)))]
    sizes = {task: split_size(args, task) for task in args.tasks}
    total = sum(sizes[c[1]] for c in pending)
    header = (f"{len(work)} run(s) planned, {len(work) - len(pending)} complete, {len(pending)} to go "
              f"({total:,} judged examples)")
    print(header)
    if args.dry_run or not pending:
        for judge, task, panel, view, method in pending:
            print(f"  {judge:<14}{task:<12}{'+'.join(panel):<46}{view:<22}{method}")
        return

    log(logfile, f"{'=' * 78}\nLAUNCH   judges={','.join(args.judges)} tasks={','.join(args.tasks)} "
                 f"methods={','.join(args.methods)} limit={args.limit or 'full'} "
                 f"context={jrun.JUDGE_CONTEXT}\n         {header}")
    out.mkdir(parents=True, exist_ok=True)
    rows, failures, done, started = [], [], 0, time.time()
    try:
        for judge in args.judges:
            batch = [c for c in pending if c[0] == judge]
            if not batch:
                continue
            left = sum(sizes[c[1]] for c in batch)
            print(f"\n{'#' * 78}\n# loading judge {judge} for {len(batch)} run(s)\n{'#' * 78}")
            log(logfile, f"JUDGE    {judge}: loading for {len(batch)} run(s), {left:,} examples")
            try:
                llm = jrun.load_judge(judge)
            except Exception as error:
                log(logfile, f"FATAL    {judge} failed to load: {type(error).__name__}: {error}")
                failures.append((judge, f"load failed: {error}"))
                traceback.print_exc()
                continue
            judged, busy = 0, 0.0
            for judge_name, task, panel, view, method in batch:
                done += 1
                label = f"{judge_name} {task} {'+'.join(panel)} {view} {method}"
                print(f"\n{'=' * 78}\n[{done}/{len(pending)}] {label}")
                log(logfile, f"START    [{done}/{len(pending)}] {label}")
                tick = time.time()
                try:
                    row = jrun.judge_once(case(args, judge_name, task, panel, view, method), llm)
                    rows.append(row)
                    judged += sizes[task]
                    busy += time.time() - tick
                    status = (f"OK       [{done}/{len(pending)}] n={row['n']} acc={row['majority_acc']:.3f} "
                              f"auroc={row.get('verbalized_auroc') or float('nan'):.3f} "
                              f"ece={row.get('verbalized_ece') or float('nan'):.3f} "
                              f"cov={row.get('verbalized_cov', 0):.3f} truncated={row['truncated']} "
                              f"prompt<={row['max_prompt_tokens']}")
                except SystemExit as error:
                    failures.append((label, f"skipped: {error}"))
                    status = f"SKIP     [{done}/{len(pending)}] {error}"
                except Exception as error:
                    traceback.print_exc()
                    failures.append((label, f"{type(error).__name__}: {error}"))
                    status = (f"CRASH    [{done}/{len(pending)}] {type(error).__name__}: {error}\n"
                              + traceback.format_exc().rstrip())
                # Rate counts only finished runs, so a crash that fails in a second cannot shorten the ETA.
                left -= sizes[task]
                eta = clock(left / (judged / busy)) if judged and busy else "?"
                log(logfile, f"{status}   {clock(time.time() - tick)}   eta this judge {eta}")
                collect(out)
            del llm
    except KeyboardInterrupt:
        log(logfile, "STOPPED  interrupted by the user")
        print("\ninterrupted")
    finally:
        summarise(logfile, args, pending, rows, failures, started)
        print(f"summary.csv -> {out / 'summary.csv'}   log -> {logfile}")


if __name__ == "__main__":
    main()
