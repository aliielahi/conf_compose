"""Debate inferences for the 15 voting groups: synchronous rounds from round 0 in the store, then candidate scoring.

The parent plans the work and spawns one child per model and stage, since vLLM frees the GPU only on exit.
Every cell, round and model is a file written once complete, so rerunning the same command resumes.
"""

import argparse
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from conf_compose.constants import ROOT, TASKS
from conf_compose.debate import (DEBATE_STORE, DebateSettings, generate_round, round0_budget, round0_cell,
                                 score_candidates, write_settings)
from conf_compose.pipelines.inference import STORE

NAME = "experiment03-debate_composition"
LOG_DIR = ROOT / "logs" / NAME
GROUPS = [
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
DEFAULT_TASKS = ["gpqa", "truthfulqa", "csqa", "gsm8k", "boolq"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, choices=sorted(TASKS))
    parser.add_argument("--groups", nargs="+", type=int, default=list(range(1, len(GROUPS) + 1)),
                        choices=range(1, len(GROUPS) + 1), metavar="1-15", help="indices into the 15 voting groups")
    parser.add_argument("--rounds", type=int, default=1, help="revision rounds after round 0")
    parser.add_argument("--stage", choices=("all", "generate", "score"), default="all")
    parser.add_argument("--limit", type=int, help="first N questions per task, in a separate cell")
    parser.add_argument("--voter", type=int, default=0)
    parser.add_argument("--match", default="_cs7s", help="round-0 cell selector, the one voting used")
    parser.add_argument("--temperature", type=float, default=0.7, help="revision decoding, as round 0")
    parser.add_argument("--context", type=int, default=32768, help="ceiling; a model's own maximum wins if lower")
    parser.add_argument("--seed", type=int, default=0, help="peer-order shuffle")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--count-tokens", action="store_true",
                        help="with --dry-run: count round-1 prompts with each model's tokenizer")
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--out-dir", type=Path, default=DEBATE_STORE)
    parser.add_argument("--log", type=Path, default=LOG_DIR / "current_run.log")
    parser.add_argument("--child", nargs=3, metavar=("STAGE", "ROUND", "MODEL"), help=argparse.SUPPRESS)
    return parser.parse_args()


def cells(args):
    return [DebateSettings(task, tuple(GROUPS[index - 1]), max_tokens=round0_budget(task, GROUPS[index - 1],
                                                                                    args.voter, args.match, args.store),
                           voter=args.voter, match=args.match, temperature=args.temperature,
                           context=args.context, limit=args.limit, seed=args.seed)
            for task in args.tasks for index in args.groups]


def owed(args, all_cells, stage, round_index, model):
    """Cells this model still owes at this stage: (ready to run, blocked on a group member's missing input)."""
    def written(settings, r):
        return all(settings.round_path(member, r, args.out_dir).exists() for member in settings.group)

    ready, blocked = [], []
    for settings in all_cells:
        if model not in settings.group:
            continue
        if stage == "generate":
            target = settings.round_path(model, round_index, args.out_dir)
            inputs = round_index == 1 or written(settings, round_index - 1)
        else:
            target = settings.scores_path(model, round_index, args.out_dir)
            inputs = all(written(settings, r) for r in range(1, round_index + 1))
        if not target.exists():
            (ready if inputs else blocked).append(settings)
    return ready, blocked


def questions(args, task):
    """Questions one turn file covers, read off round 0 without loading it."""
    path = round0_cell(task, GROUPS[0][0], args.voter, args.match, store=args.store) / "test.jsonl"
    count = sum(1 for line in path.open() if line.strip())
    return min(count, args.limit) if args.limit else count


def steps(args):
    stages = (["generate"] if args.stage in ("all", "generate") else []) + \
             (["score"] if args.stage in ("all", "score") else [])
    plan = []
    for stage in stages:
        rounds = range(1, args.rounds + 1) if stage == "generate" else [args.rounds]
        for round_index in rounds:
            for model in sorted({member for index in args.groups for member in GROUPS[index - 1]}):
                plan.append((stage, round_index, model))
    return plan


def native_context(model: str) -> int:
    """The model's own maximum length, read from its config; gemma-2 stops at 8192."""
    from transformers import AutoConfig

    from conf_compose.utils.llm_calls.hf_models import HF_models

    config = AutoConfig.from_pretrained(HF_models[model])
    text = getattr(config, "text_config", None) or config
    return int(getattr(text, "max_position_embeddings"))


def run_child(args):
    """One loaded model does every cell it owes for one stage and round, then exits to free the GPU."""
    from conf_compose.pipelines.inference import load_model

    stage, round_index, model = args.child[0], int(args.child[1]), args.child[2]
    jobs, _ = owed(args, cells(args), stage, round_index, model)
    if not jobs:
        print(f"{model}: nothing pending for {stage} round {round_index}")
        return
    context = min(args.context, native_context(model))
    start = time.time()
    llm = load_model(f"vllm/{model}", max_model_len=context)
    print(f"{model}: loaded in {time.time() - start:.0f}s, context {context}, {len(jobs)} cell(s)", flush=True)
    if stage == "generate":
        written = generate_round(llm, model, jobs, round_index, context, args.store, args.out_dir)
    else:
        written = score_candidates(llm, model, jobs, round_index, args.store, args.out_dir)
    print(f"{model}: wrote {sum(written.values())} row(s) in {len(written)} cell(s) "
          f"in {time.time() - start:.0f}s", flush=True)


def count_tokens(args, all_cells):
    """Round-1 prompt lengths with each model's real tokenizer and chat template, before any GPU time."""
    from transformers import AutoTokenizer

    from conf_compose.data import get_task
    from conf_compose.debate.engine import MIN_OUTPUT, build_turns
    from conf_compose.utils.llm_calls.hf_models import CHAT_TEMPLATE_KWARGS, HF_models

    print(f"\n{'task':<12}{'model':<10}{'context':>8}{'budget':>8}{'longest':>9}{'capped':>8}{'overflow':>9}")
    for model in sorted({member for settings in all_cells for member in settings.group}):
        tokenizer = AutoTokenizer.from_pretrained(HF_models[model])
        context = min(args.context, native_context(model))
        for task_name in args.tasks:
            task = get_task(task_name)
            jobs = [s for s in all_cells if s.task == task_name and model in s.group]
            turns = [turn for settings in jobs
                     for turn in build_turns(task, settings, model, 1, args.store, args.out_dir)]
            # Rendered to text, then encoded without special tokens: exactly how the engine counts and sends it.
            texts = [tokenizer.apply_chat_template(turn["messages"], tokenize=False, add_generation_prompt=True,
                                                   **CHAT_TEMPLATE_KWARGS.get(model, {})) for turn in turns]
            lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]] if texts else []
            budget = jobs[0].max_tokens
            capped = sum(context - n < budget for n in lengths)
            overflow = sum(context - n < MIN_OUTPUT for n in lengths)
            print(f"{task_name:<12}{model:<10}{context:>8}{budget:>8}{max(lengths, default=0):>9}"
                  f"{capped:>8}{overflow:>9}", flush=True)


def log(path: Path, message: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}\n")


def main():
    args = parse_args()
    args.store, args.out_dir = args.store.resolve(), args.out_dir.resolve()
    if args.child:
        # vLLM's background threads can keep a failed child alive forever; exit hard so the parent sees it.
        try:
            run_child(args)
        except BaseException:
            traceback.print_exc()
            sys.stdout.flush()
            os._exit(1)
        sys.stdout.flush()
        os._exit(0)
    all_cells = cells(args)
    plan = steps(args)
    sizes = {task: questions(args, task) for task in args.tasks}
    work = {(stage, r, model): sum(sizes[s.task] for s in all_cells if model in s.group) for stage, r, model in plan}
    total = sum(work.values())
    header = (f"{len(all_cells)} cell(s): {len(args.tasks)} task(s) x {len(args.groups)} group(s), "
              f"{args.rounds} round(s), stage {args.stage}, limit {args.limit or 'full'}; "
              f"{len(plan)} model step(s), {total:,} agent turn(s)")
    print(header)
    if args.dry_run:
        for (stage, r, model), turns in work.items():
            print(f"  {stage:<9}round {r}  {model:<9}{turns:>8} turn(s)")
        if args.count_tokens:
            count_tokens(args, all_cells)
        return

    for settings in all_cells:
        write_settings(settings, args.store, args.out_dir)
    log(args.log, f"{'=' * 78}\nLAUNCH   {header}")
    started, executed, busy, reached = time.time(), 0, 0.0, 0
    outcome = {"ok": [], "done": [], "blocked": [], "failed": []}
    left = total
    try:
        for index, (stage, r, model) in enumerate(plan, 1):
            label = f"[{index}/{len(plan)}] {stage} round {r} {model}"
            reached = index
            ready, blocked = owed(args, all_cells, stage, r, model)
            left -= work[(stage, r, model)]
            if blocked:
                outcome["blocked"].append(f"{label}: {len(blocked)} cell(s) wait on a failed input")
            if not ready:
                if not blocked:
                    outcome["done"].append(label)
                log(args.log, f"SKIP     {label}: " + (f"{len(blocked)} cell(s) wait on a failed input"
                                                       if blocked else "already written"))
                continue
            child_log = LOG_DIR / f"{stage}_round{r}_{model}.log"
            note = f", {len(blocked)} more blocked" if blocked else ""
            log(args.log, f"START    {label} ({len(ready)} cell(s){note}) -> {child_log}")
            tick = time.time()
            code = spawn(args, stage, r, model, child_log)
            seconds = time.time() - tick
            # The rate counts only steps that ran, so skipped work cannot shorten the estimate.
            executed += work[(stage, r, model)]
            busy += seconds
            eta = timedelta(seconds=int(left * busy / max(executed, 1)))
            if code == 0:
                outcome["ok"].append(label)
                log(args.log, f"OK       {label} {timedelta(seconds=int(seconds))}   eta {eta}")
            else:
                outcome["failed"].append(f"{label}: exit {code}, see {child_log}")
                tail = "".join(child_log.read_text().splitlines(True)[-15:])
                log(args.log, f"CRASH    {label}: exit {code}; tail of {child_log}:\n{tail}")
    except KeyboardInterrupt:
        log(args.log, "STOPPED  interrupted by the user")
    except Exception:
        log(args.log, "FATAL    " + traceback.format_exc())
        raise
    finally:
        lines = [f"SUMMARY  {len(outcome['ok'])} ran, {len(outcome['done'])} already written, "
                 f"{len(outcome['blocked'])} with blocked cells, {len(outcome['failed'])} failed, "
                 f"{len(plan) - reached} not reached, in {timedelta(seconds=int(time.time() - started))}"]
        lines += [f"  FAILED  {failure}" for failure in outcome["failed"]]
        lines += [f"  BLOCKED {step}" for step in outcome["blocked"]]
        log(args.log, "\n".join(lines) + f"\n{'-' * 78}")
        print("\n".join(lines))


def spawn(args, stage, round_index, model, child_log):
    """One child per model and stage; its output goes to its own log in this experiment's log folder."""
    command = [sys.executable, __file__, "--child", stage, str(round_index), model, "--tasks", *args.tasks,
               "--groups", *map(str, args.groups), "--rounds", str(args.rounds), "--voter", str(args.voter),
               "--match", args.match, "--temperature", str(args.temperature), "--context", str(args.context),
               "--seed", str(args.seed), "--store", str(args.store), "--out-dir", str(args.out_dir)]
    command += ["--limit", str(args.limit)] if args.limit else []
    child_log.parent.mkdir(parents=True, exist_ok=True)
    with child_log.open("a") as handle:
        return subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode


if __name__ == "__main__":
    main()
