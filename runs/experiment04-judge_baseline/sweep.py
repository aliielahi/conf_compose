"""Sweep the judge over tasks, panels and judges, showing one confidence estimator at a time."""

import argparse
import csv
import importlib.util
from argparse import Namespace
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


def main():
    args = parse_args()
    work = plan(args)
    out = Path(args.out_dir)
    pending = [c for c in work
               if args.force or not (out / c[1] / jrun._cell(case(args, *c)) / "verdicts.jsonl").exists()]
    print(f"{len(work)} run(s) planned, {len(work) - len(pending)} already on disk, {len(pending)} to go")
    print(f"judges={len(args.judges)} tasks={len(args.tasks)} panels={len(set(map(str, [c[2] for c in work])))} "
          f"arms={'' if args.no_control else 'control+'}{len(args.methods)} method(s)")
    if args.dry_run or not pending:
        for judge, task, panel, view, method in pending:
            print(f"  {judge:<14}{task:<12}{'+'.join(panel):<46}{view:<22}{method}")
        return

    out.mkdir(parents=True, exist_ok=True)
    rows, done = [], 0
    for judge in args.judges:
        batch = [c for c in pending if c[0] == judge]
        if not batch:
            continue
        print(f"\n{'#' * 78}\n# loading judge {judge} for {len(batch)} run(s)\n{'#' * 78}")
        llm = load_model(judge)
        for judge_name, task, panel, view, method in batch:
            done += 1
            print(f"\n{'=' * 78}\n[{done}/{len(pending)}] {judge_name} {task} "
                  f"{'+'.join(panel)} {view} {method}")
            try:
                rows.append(jrun.judge_once(case(args, judge_name, task, panel, view, method), llm))
            except SystemExit as error:
                print(f"SKIPPED: {error}")
            write(out / "summary.csv", rows)
        del llm
    print(f"\n{len(rows)}/{len(pending)} run(s) completed -> {out / 'summary.csv'}")


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
