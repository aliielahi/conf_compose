"""Teacher-forced token log-probabilities of one scorer model on every ProcessBench solution."""

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

from data.processbench import SPLITS, load

RUNS = Path(__file__).resolve().parents[2] / "runs" / "cross_scoring"
MAX_TOKENS = 8192


def encode(tokenizer, problem, steps):
    """Chat-formatted token ids and the step index of every token (-1 outside the solution)."""
    solution = "\n\n".join(steps)
    text = tokenizer.apply_chat_template([{"role": "user", "content": problem},
                                          {"role": "assistant", "content": solution}], tokenize=False)
    start = text.rindex(solution)
    bounds = np.cumsum([0] + [len(s) + 2 for s in steps[:-1]])
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    step = np.full(len(encoded.input_ids), -1)
    for i, (begin, _) in enumerate(encoded.offset_mapping):
        if 0 <= begin - start < len(solution):
            step[i] = np.searchsorted(bounds, begin - start, side="right") - 1
    return encoded.input_ids, step


def score(llm, tokenizer, rows, limit):
    """Flat arrays over all solution tokens (item, step, log-prob of the token, of the top token) and scored items."""
    encoded = [encode(tokenizer, row["problem"], row["steps"]) for row in rows]
    keep = [i for i, (ids, _) in enumerate(encoded) if len(ids) < limit]
    outputs = llm.generate([{"prompt_token_ids": encoded[i][0]} for i in keep],
                           SamplingParams(max_tokens=1, prompt_logprobs=1))
    columns = {"item": [], "step": [], "actual": [], "top": []}
    for i, output in zip(keep, outputs):
        ids, step = encoded[i]
        for t in np.flatnonzero(step >= 0):
            candidates = output.prompt_logprobs[t]
            columns["item"].append(i)
            columns["step"].append(step[t])
            columns["actual"].append(candidates[ids[t]].logprob)
            columns["top"].append(max(c.logprob for c in candidates.values()))
    columns["scored"] = keep
    return {k: np.asarray(v, dtype=np.float32 if k in ("actual", "top") else np.int32) for k, v in columns.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    args = parser.parse_args()
    limit = min(MAX_TOKENS, getattr(AutoConfig.from_pretrained(args.model), "max_position_embeddings", MAX_TOKENS))
    llm = LLM(args.model, max_model_len=limit, gpu_memory_utilization=0.85, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    out = RUNS / "scores" / args.model.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        rows = load(split)
        np.savez(out / f"{split}.npz", labels=np.array([r["label"] for r in rows]),
                 n_steps=np.array([len(r["steps"]) for r in rows]), **score(llm, tokenizer, rows, limit))
        print(f"{args.model} {split}: {len(rows)} solutions", flush=True)


if __name__ == "__main__":
    main()
