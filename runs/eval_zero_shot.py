"""Evaluate zero-shot correctness and confidence quality for one model on one task (validation + test)."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import get_task
from pipelines import ZeroShotConfig, run_zero_shot
from pipelines.report import confidence_report, format_report
from utils.llm_calls import LLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vllm/q3-4bi")
    parser.add_argument("--task", default="gsm8k")
    parser.add_argument("--n-val", type=int, default=0, help="0 skips calibration")
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--scopes", nargs="+", default=["response"])
    parser.add_argument("--tail-fraction", type=float, default=0.1)
    parser.add_argument("--debias", action="store_true")
    parser.add_argument("--no-verbalized", action="store_true")
    parser.add_argument("--no-verification", action="store_true")
    parser.add_argument("--verbal-temperature", type=float, default=1.0)
    parser.add_argument("--consistency-temperatures", type=float, nargs="*", default=[0.7])
    parser.add_argument("--consistency-samples", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--calibrator", choices=["beta", "platt"], default="beta")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32, help="hf backend only")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vllm backend only")
    parser.add_argument("--out-dir", default="runs/outputs")
    parser.add_argument("--cache-dir", default="runs/cache")
    args = parser.parse_args()

    config = ZeroShotConfig(
        max_tokens=args.max_tokens, scopes=tuple(args.scopes), tail_fraction=args.tail_fraction, debias=args.debias,
        verbalized=not args.no_verbalized, verbal_temperature=args.verbal_temperature,
        verification=not args.no_verification, consistency_temperatures=tuple(args.consistency_temperatures),
        consistency_samples=args.consistency_samples, top_p=args.top_p, top_k=args.top_k,
    )

    start = time.time()
    task = get_task(args.task)
    splits = {"validation": task.load("validation", n=args.n_val) if args.n_val else [],
              "test": task.load("test", n=args.n_test)}
    backend_kwargs = ({"batch_size": args.batch_size} if args.model.startswith("hf/")
                      else {"gpu_memory_utilization": args.gpu_memory_utilization})
    llm = LLM(args.model, cache_dir=args.cache_dir, **backend_kwargs)
    print(f"{llm} | {task.name} val={len(splits['validation'])} test={len(splits['test'])} "
          f"| loaded in {time.time() - start:.0f}s")

    combined = run_zero_shot(llm, task, splits["validation"] + splits["test"], config)
    n_val = len(splits["validation"])
    records = {split: rows for split, rows in (("validation", combined[:n_val]), ("test", combined[n_val:])) if rows}
    report = confidence_report(records.get("validation", []), records["test"], args.calibrator, args.n_boot)

    for split, rows in records.items():
        accuracy = sum(r["correct"] for r in rows) / len(rows)
        truncated = sum(r["finish_reason"] == "length" for r in rows)
        implicit = sum(r["prediction"] is not None and not r["explicit_answer"] for r in rows)
        print(f"{split}: accuracy={accuracy:.3f} truncated={truncated} implicit_answers={implicit}")
    calibration = f"{args.calibrator} calibration fitted on validation" if n_val else "no calibration"
    print(f"\ntest confidence quality ({calibration}) | {time.time() - start:.0f}s\n")
    print("\n".join(format_report(report)))

    out_dir = Path(args.out_dir) / f"{task.name}_{llm.model.replace('/', '__')}_val{args.n_val}_test{args.n_test}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        (out_dir / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out_dir / "report.json").write_text(json.dumps({"args": vars(args), "report": report}, indent=2))
    print(f"\nsaved {out_dir}")


if __name__ == "__main__":
    main()
