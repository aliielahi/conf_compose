"""Evaluate zero-shot correctness and confidence quality for one model on one task; defaults from constants.json."""

import argparse
import json
import time

from conf_compose.constants import (CACHE_DIR, EVALUATION, HF, RESULTS_DIR, SAMPLING, SEQUENCE_PROBABILITY, TASKS,
                                    VLLM)
from conf_compose.data import get_task
from conf_compose.pipelines import ZeroShotConfig, run_zero_shot
from conf_compose.pipelines.report import confidence_report, format_report
from conf_compose.pipelines.runs import run_dir, split_summary
from conf_compose.utils.llm_calls import LLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--n-val", type=int, help="default from constants; 0 skips calibration")
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--scopes", nargs="+", default=SEQUENCE_PROBABILITY["scopes"])
    parser.add_argument("--tail-fraction", type=float, default=SEQUENCE_PROBABILITY["tail_fraction"])
    parser.add_argument("--debias", action="store_true")
    parser.add_argument("--no-verbalized", action="store_true")
    parser.add_argument("--no-verification", action="store_true")
    parser.add_argument("--verbal-temperature", type=float, default=SAMPLING["verbal_temperature"])
    parser.add_argument("--consistency-temperatures", type=float, nargs="*",
                        default=[SAMPLING["consistency_temperature"]])
    parser.add_argument("--consistency-samples", type=int, default=SAMPLING["consistency_samples"])
    parser.add_argument("--top-p", type=float, default=SAMPLING["top_p"])
    parser.add_argument("--top-k", type=int, default=SAMPLING["top_k"])
    parser.add_argument("--calibrator", choices=["beta", "platt"], default=EVALUATION["calibrator"])
    parser.add_argument("--n-boot", type=int, default=EVALUATION["n_boot"])
    parser.add_argument("--batch-size", type=int, default=HF["batch_size"], help="hf backend only")
    parser.add_argument("--gpu-memory-utilization", type=float, default=VLLM["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=VLLM["max_model_len"])
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    args = parser.parse_args()
    task_defaults = TASKS[args.task]
    args.n_val = task_defaults["n_val"] if args.n_val is None else args.n_val
    args.n_test = task_defaults["n_test"] if args.n_test is None else args.n_test
    args.max_tokens = args.max_tokens or task_defaults["max_tokens"]
    return args


def main():
    args = parse_args()
    config = ZeroShotConfig(
        max_tokens=args.max_tokens, scopes=tuple(args.scopes), tail_fraction=args.tail_fraction, debias=args.debias,
        verbalized=not args.no_verbalized, verbal_temperature=args.verbal_temperature,
        verification=not args.no_verification, consistency_temperatures=tuple(args.consistency_temperatures),
        consistency_samples=args.consistency_samples, top_p=args.top_p, top_k=args.top_k,
    )

    start = time.time()
    task = get_task(args.task)
    validation = task.load("validation", n=args.n_val) if args.n_val else []
    test = task.load("test", n=args.n_test)
    backend_kwargs = ({"batch_size": args.batch_size} if args.model.startswith("hf/") else
                      {"gpu_memory_utilization": args.gpu_memory_utilization, "max_model_len": args.max_model_len})
    llm = LLM(args.model, cache_dir=args.cache_dir, **backend_kwargs)
    print(f"{llm} | {task.name} val={len(validation)} test={len(test)} max_tokens={args.max_tokens} "
          f"| loaded in {time.time() - start:.0f}s")

    combined = run_zero_shot(llm, task, validation + test, config)
    records = {"validation": combined[:len(validation)], "test": combined[len(validation):]}
    records = {split: rows for split, rows in records.items() if rows}
    report = confidence_report(records.get("validation", []), records["test"], args.calibrator, args.n_boot)
    summaries = {split: split_summary(rows) for split, rows in records.items()}

    for split, summary in summaries.items():
        print(f"{split}: " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                      for k, v in summary.items()))
    calibration = f"{args.calibrator} calibration fitted on validation" if validation else "no calibration"
    print(f"\ntest confidence quality ({calibration}) | {time.time() - start:.0f}s\n")
    print("\n".join(format_report(report)))

    out_dir = run_dir(args.out_dir, task.name, args.model, args.n_val, args.n_test)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        (out_dir / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    payload = {"args": vars(args), "config": config.to_dict(), "splits": summaries, "report": report}
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out_dir}")


if __name__ == "__main__":
    main()
