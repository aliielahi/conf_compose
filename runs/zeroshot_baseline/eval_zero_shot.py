"""Evaluate zero-shot correctness and confidence quality for one model on one or more tasks (model loaded once)."""

import argparse
import json
import time

from conf_compose.constants import (CACHE_DIR, EVALUATION, HF, RESULTS_DIR, SAMPLING, SEQUENCE_PROBABILITY, TASKS,
                                    VLLM)
from conf_compose.data import get_task
from conf_compose.pipelines import ZeroShotConfig, run_zero_shot
from conf_compose.pipelines.report import confidence_report, format_report
from conf_compose.pipelines.runs import header_lines, run_dir, split_summary, timing_line
from conf_compose.utils.llm_calls import LLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", nargs="+", required=True, choices=sorted(TASKS))
    parser.add_argument("--n-val", type=int, help="default per task from constants; 0 skips calibration")
    parser.add_argument("--n-test", type=int, help="default per task from constants")
    parser.add_argument("--max-tokens", type=int, help="default per task from constants")
    parser.add_argument("--scopes", nargs="+", default=SEQUENCE_PROBABILITY["scopes"])
    parser.add_argument("--tail-fraction", type=float, default=SEQUENCE_PROBABILITY["tail_fraction"])
    parser.add_argument("--debias", action="store_true")
    parser.add_argument("--no-verbalized", action="store_true")
    parser.add_argument("--no-verification", action="store_true")
    parser.add_argument("--verbal-temperature", type=float, default=SAMPLING["verbal_temperature"])
    parser.add_argument("--consistency-temperatures", type=float, nargs="*",
                        default=[SAMPLING["consistency_temperature"]])
    parser.add_argument("--consistency-samples", type=int, default=SAMPLING["consistency_samples"])
    parser.add_argument("--answer-temperature", type=float, default=0.0,
                        help="0 keeps greedy answers; >0 makes each voter generate its own answer")
    parser.add_argument("--voters", type=int, default=1, help="independent voters per model, written as _rep<i>")
    parser.add_argument("--execution", default="", help="sampling identity; each voter appends its own index")
    parser.add_argument("--top-p", type=float, default=SAMPLING["top_p"])
    parser.add_argument("--top-k", type=int, default=SAMPLING["top_k"])
    parser.add_argument("--calibrator", choices=["beta", "platt"], default=EVALUATION["calibrator"])
    parser.add_argument("--n-boot", type=int, default=EVALUATION["n_boot"])
    parser.add_argument("--batch-size", type=int, default=HF["batch_size"], help="hf backend only")
    parser.add_argument("--gpu-memory-utilization", type=float, default=VLLM["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=VLLM["max_model_len"])
    parser.add_argument("--include-unanswered", action="store_true",
                        help="score examples with no extracted answer too (counted as confidence 0)")
    parser.add_argument("--skip-existing", action="store_true", help="skip tasks whose report.json exists")
    parser.add_argument("--verbose", action="store_true", help="show vLLM engine logs")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    return parser.parse_args()


def task_settings(args, task_name):
    defaults = TASKS[task_name]
    return {
        "n_val": defaults["n_val"] if args.n_val is None else args.n_val,
        "n_test": defaults["n_test"] if args.n_test is None else args.n_test,
        "max_tokens": args.max_tokens or defaults["max_tokens"],
    }


def load_model(args):
    if args.model.startswith("hf/"):
        return LLM(args.model, cache_dir=args.cache_dir, batch_size=args.batch_size)
    engine = {"max_num_seqs": VLLM["max_num_seqs"], "max_num_batched_tokens": VLLM["max_num_batched_tokens"]}
    return LLM(args.model, cache_dir=args.cache_dir, gpu_memory_utilization=args.gpu_memory_utilization,
               max_model_len=args.max_model_len, quiet=not args.verbose, engine_kwargs=engine)


def evaluate_task(llm, args, task_name, voter=None):
    settings = task_settings(args, task_name)
    suffix = "" if voter is None else f"_rep{voter}"
    out_dir = run_dir(args.out_dir, task_name, args.model, settings["n_val"], settings["n_test"], suffix)
    if args.skip_existing and (out_dir / "report.json").exists():
        print(f"skip {task_name}{suffix}: {out_dir} exists")
        return
    llm.execution = args.execution if voter is None else f"{args.execution}:voter{voter}"

    start = time.time()
    config = ZeroShotConfig(
        max_tokens=settings["max_tokens"], scopes=tuple(args.scopes), tail_fraction=args.tail_fraction,
        answer_temperature=args.answer_temperature, debias=args.debias,
        verbalized=not args.no_verbalized, verbal_temperature=args.verbal_temperature,
        verification=not args.no_verification, consistency_temperatures=tuple(args.consistency_temperatures),
        consistency_samples=args.consistency_samples, top_p=args.top_p, top_k=args.top_k,
    )
    task = get_task(task_name)
    validation = task.load("validation", n=settings["n_val"]) if settings["n_val"] else []
    test = task.load("test", n=settings["n_test"])
    print(f"\n{llm} | {task_name}{suffix} val={len(validation)} test={len(test)} "
          f"max_tokens={settings['max_tokens']} answer_t={args.answer_temperature} execution={llm.execution!r}")

    timings = {}
    combined = run_zero_shot(llm, task, validation + test, config, timings)
    records = {"validation": combined[:len(validation)], "test": combined[len(validation):]}
    records = {split: rows for split, rows in records.items() if rows}
    report = confidence_report(records.get("validation", []), records["test"], args.calibrator, args.n_boot,
                               answered_only=not args.include_unanswered)
    summaries = {split: split_summary(rows) for split, rows in records.items()}

    print(timing_line(timings))
    print("\n".join(header_lines(summaries, records, answered_only=not args.include_unanswered)))
    print(f"\ntest confidence quality ({args.calibrator} calibration on validation) | {time.time() - start:.0f}s\n")
    print("\n".join(format_report(report)))

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        (out_dir / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    payload = {"model": args.model, "task": task_name, "voter": voter, "execution": llm.execution,
               "settings": settings, "config": config.to_dict(), "args": vars(args), "splits": summaries,
               "timings": timings, "report": report}
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out_dir}")


def main():
    args = parse_args()
    start = time.time()
    llm = load_model(args)
    print(f"loaded {llm} in {time.time() - start:.0f}s")
    for task_name in args.tasks:
        if args.voters <= 1 and args.answer_temperature == 0:
            evaluate_task(llm, args, task_name)
            continue
        for voter in range(args.voters):
            evaluate_task(llm, args, task_name, voter)


if __name__ == "__main__":
    main()
