"""Smoke test: answer a few reasoning questions and score them with both confidence estimators."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confidence_estimators import SequenceProbability, VerbalizedConfidence
from data import get_task
from utils.llm_calls import LLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hf/q3-4bi")
    parser.add_argument("--task", default="gsm8k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    start = time.time()
    task = get_task(args.task)
    examples = task.load(args.split, n=args.n)
    llm = LLM(args.model)
    print(f"{llm} | {task.name}/{args.split} n={len(examples)} | loaded in {time.time() - start:.1f}s")

    prompts = [task.prompt(ex) for ex in examples]
    responses = llm(prompts, max_tokens=args.max_tokens, temperature=0.0)
    answers = [task.extract_answer(r) for r in responses]
    predictions = [a.text if a else None for a in answers]
    correct = [task.is_correct(p, ex) for p, ex in zip(predictions, examples)]

    sequence = {scope: SequenceProbability(llm, scope=scope).estimate(task, examples, responses)
                for scope in ("answer", "answer_no_reasoning", "response")}
    verbalized = VerbalizedConfidence(llm).estimate(prompts, responses)

    records = []
    for i, example in enumerate(examples):
        record = {
            "id": example.id,
            "gold": example.answer,
            "prediction": predictions[i],
            "correct": correct[i],
            "verbalized": verbalized[i].confidence,
            "verbalized_raw": verbalized[i].raw,
            **{f"seq_{scope}": vars(results[i]) if results[i] else None for scope, results in sequence.items()},
            "response": responses[i],
        }
        records.append(record)
        print(f"\n=== {example.id} ===")
        print(f"Q: {example.question}")
        print(f"A: {responses[i]}")
        print(f"gold={example.answer} pred={predictions[i]} correct={correct[i]}")
        print(f"verbalized={_fmt(verbalized[i].confidence)} raw={verbalized[i].raw}")
        for scope, results in sequence.items():
            r = results[i]
            if r is None:
                print(f"seq[{scope}] n/a")
            else:
                print(f"seq[{scope}] conf={_fmt(r.confidence)} debiased={_fmt(r.debiased_confidence)} "
                      f"mean_lp={r.mean_logprob:.3f} null_lp={_fmt(r.null_mean_logprob)} tokens={r.n_tokens}")

    print(f"\naccuracy={sum(correct)}/{len(correct)} | done in {time.time() - start:.1f}s")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(json.dumps(r) for r in records) + "\n")
        print(f"saved {args.out}")


def _fmt(value):
    return "None" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
