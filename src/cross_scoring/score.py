"""Teacher-forced token log-probabilities of one scorer model on every ProcessBench solution."""

import argparse
from pathlib import Path

import numpy as np

from cross_scoring.teacher_forcing import encode, load, logprobs
from data.processbench import SPLITS
from data.processbench import load as load_split

RUNS = Path(__file__).resolve().parents[2] / "runs" / "cross_scoring"


def score(llm, tokenizer, rows, limit):
    """Flat arrays over all solution tokens (item, step, log-prob of the token, of the top token) and scored items."""
    encoded = []
    for row in rows:
        starts = np.cumsum([0] + [len(s) + 2 for s in row["steps"][:-1]])
        encoded.append(encode(tokenizer, row["problem"], "\n\n".join(row["steps"]), starts))
    keep = [i for i, (ids, _) in enumerate(encoded) if len(ids) < limit]
    wanted = [np.flatnonzero(encoded[i][1] >= 0) for i in keep]
    scored = logprobs(llm, [encoded[i][0] for i in keep], wanted)
    columns = {"item": [], "step": [], "actual": [], "top": []}
    for i, positions, (actual, top) in zip(keep, wanted, scored):
        columns["item"].append(np.full(len(positions), i))
        columns["step"].append(encoded[i][1][positions])
        columns["actual"].append(actual)
        columns["top"].append(top)
    out = {k: np.concatenate(v).astype(np.float32 if k in ("actual", "top") else np.int32) for k, v in columns.items()}
    return {**out, "scored": np.array(keep, dtype=np.int32)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    args = parser.parse_args()
    out = RUNS / "scores" / args.model.replace("/", "__")
    if all((out / f"{split}.npz").exists() for split in args.splits):
        return
    llm, tokenizer, limit = load(args.model)
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        if (out / f"{split}.npz").exists():
            continue
        rows = load_split(split)
        np.savez(out / f"{split}.npz", labels=np.array([r["label"] for r in rows]),
                 n_steps=np.array([len(r["steps"]) for r in rows]), **score(llm, tokenizer, rows, limit))
        print(f"{args.model} {split}: {len(rows)} solutions", flush=True)


if __name__ == "__main__":
    main()
