"""Evaluate zero-shot correctness and confidence quality for one model on one task."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import get_task
from pipelines import run_zero_shot, signal, signal_names
from utils.llm_calls import LLM
from utils.metrics import summarize

COLUMNS = ("accuracy", "mean_conf", "ece", "ace", "brier", "nll", "auroc", "auarc", "oracle_auarc")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hf/q3-4bi")
    parser.add_argument("--task", default="gsm8k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--scopes", nargs="+", default=["answer", "answer_no_reasoning", "response"])
    parser.add_argument("--no-verbalized", action="store_true")
    parser.add_argument("--out-dir", default="runs/outputs")
    parser.add_argument("--cache-dir", default="runs/cache")
    args = parser.parse_args()

    start = time.time()
    task = get_task(args.task)
    examples = task.load(args.split, n=args.n)
    llm = LLM(args.model, batch_size=args.batch_size, cache_dir=args.cache_dir)
    print(f"{llm} | {task.name}/{args.split} n={len(examples)} | loaded in {time.time() - start:.0f}s")

    records = run_zero_shot(llm, task, examples, max_tokens=args.max_tokens, scopes=args.scopes,
                            verbalized=not args.no_verbalized)
    correct = [record["correct"] for record in records]
    extracted = sum(record["prediction"] is not None for record in records)

    metrics = {}
    for name in signal_names(records):
        metrics[name] = summarize(signal(records, name), correct)
        metrics[name]["missing"] = sum(record["confidence"][name] is None for record in records)

    print(f"\naccuracy={sum(correct) / len(correct):.3f} extracted={extracted}/{len(records)} "
          f"| {time.time() - start:.0f}s\n")
    print(f"{'signal':<30}" + "".join(f"{c:>13}" for c in COLUMNS) + f"{'missing':>9}")
    for name, row in metrics.items():
        print(f"{name:<30}" + "".join(f"{row[c]:>13.4f}" for c in COLUMNS) + f"{row['missing']:>9}")

    out_dir = Path(args.out_dir) / f"{task.name}_{args.split}_{llm.model.replace('/', '__')}_n{len(records)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "records.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    (out_dir / "metrics.json").write_text(json.dumps({"args": vars(args), "metrics": metrics}, indent=2))
    print(f"\nsaved {out_dir}")


if __name__ == "__main__":
    main()
