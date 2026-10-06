"""Sweep the judge over tasks, panels and one shown confidence at a time, loading the judge once."""

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

# Panels the judge is never a member of, so it never rates its own answer.
PANELS = {
    "qwen_homog": ["vllm/q3-4bi", "vllm/q3-8bi"],
    "llama_homog": ["vllm/l32-3bi", "vllm/l31-8bi"],
    "hetero_pair": ["vllm/q3-4bi", "vllm/l31-8bi"],
    "hetero_triple": ["vllm/q3-4bi", "vllm/l31-8bi", "vllm/g2-9i"],
}
TASKS = ["csqa", "boolq", "gsm8k", "truthfulqa", "gpqa"]
METHODS = ["consistency_t0.7", "seq_response_debiased"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", default="vllm/g3-12i")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--panels", nargs="+", default=list(PANELS), choices=list(PANELS))
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        help="each is a separate run; the judge never sees two at once")
    parser.add_argument("--modes", nargs="+", default=["verbalized"])
    parser.add_argument("--voter", type=int, default=0)
    parser.add_argument("--match", default="_cs7s")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, help="first N examples; omit for the whole split")
    parser.add_argument("--word-limit", type=int, default=150)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / NAME))
    return parser.parse_args()


def cases(args):
    """reasoning is the control; each method is one more run with that signal alone shown."""
    for task in args.tasks:
        for name in args.panels:
            yield task, name, "reasoning", args.methods[0]
            for method in args.methods:
                yield task, name, "reasoning_confidence", method


def main():
    args = parse_args()
    plan = list(cases(args))
    print(f"{len(plan)} run(s): {len(args.tasks)} task(s) x {len(args.panels)} panel(s) "
          f"x (1 control + {len(args.methods)} method(s)), judge={args.judge}")

    llm, rows = load_model(args.judge), []
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for index, (task, panel, view, method) in enumerate(plan, 1):
        print(f"\n{'=' * 78}\n[{index}/{len(plan)}] {task} {panel} {view} {method}")
        one = Namespace(**{**vars(args), "task": task, "panel": PANELS[panel], "view": view,
                           "confidence_method": method})
        try:
            rows.append({"panel_name": panel, **jrun.judge_once(one, llm)})
        except SystemExit as error:
            print(f"SKIPPED {task}/{panel}: {error}")
        write(out / "summary.csv", rows)
    print(f"\n{len(rows)}/{len(plan)} run(s) completed -> {out / 'summary.csv'}")


def write(path, rows):
    """Rewritten after every run, so a crash still leaves the finished rows behind."""
    if not rows:
        return
    fields = list({key: None for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
