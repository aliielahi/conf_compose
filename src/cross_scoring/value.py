"""A scorer's belief in the panel's answer after each prefix of every ProcessBench solution.

belief_k = log P(answer | problem, first k steps, "Therefore, the final answer is $\\boxed{"), teacher-forced; the
answer is the most common final answer of the panel's own greedy solutions. A wrong step should make it drop.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from cross_scoring.solve import SPLITS
from cross_scoring.teacher_forcing import encode, load, logprobs
from data.answers import normalise
from data.processbench import load as load_split

RUNS = Path(__file__).resolve().parents[2] / "runs" / "cross_scoring"
CUE = "Therefore, the final answer is $\\boxed"


def consensus(split):
    """Per problem, the most common final answer over all scorers' greedy solutions."""
    answers = [json.loads(p.read_text()) for p in sorted((RUNS / "answers").glob(f"*/{split}.json"))]
    out = {}
    for problem in answers[0]:
        votes = Counter(normalise(a[problem]) for a in answers if a.get(problem) is not None)
        if votes:
            out[problem] = next(a[problem] for a in answers if normalise(a.get(problem)) == votes.most_common(1)[0][0])
    return out


def beliefs(llm, tokenizer, rows, targets, limit):
    """Flat arrays (item, k, belief) over every solution with a target and every prefix length k = 0..steps."""
    prompts, wanted, keys = [], [], []
    for i, row in enumerate(rows):
        if row["problem"] not in targets:
            continue
        for k in range(len(row["steps"]) + 1):
            prefix = "".join(f"{s}\n\n" for s in row["steps"][:k]) + CUE
            response = prefix + "{" + targets[row["problem"]] + "}$"
            ids, span = encode(tokenizer, row["problem"], response, [0, len(prefix), len(response) - 1])
            if len(ids) < limit:
                prompts.append(ids)
                wanted.append(np.flatnonzero(span == 1))
                keys.append((i, k))
    scored = logprobs(llm, prompts, wanted)
    item, k = np.array(keys, dtype=np.int32).T
    return {"item": item, "k": k, "belief": np.array([actual.sum() for actual, _ in scored], dtype=np.float32)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    args = parser.parse_args()
    out = RUNS / "beliefs" / args.model.replace("/", "__")
    if all((out / f"{split}.npz").exists() for split in args.splits):
        return
    llm, tokenizer, limit = load(args.model)
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        if (out / f"{split}.npz").exists():
            continue
        rows = load_split(split)
        np.savez(out / f"{split}.npz", **beliefs(llm, tokenizer, rows, consensus(split), limit))
        print(f"{args.model} {split}: beliefs of {len(rows)} solutions", flush=True)


if __name__ == "__main__":
    main()
