"""Sampled reasoning traces, and at cut points of some traces the forced-answer belief and the rollout value.

At each cut of a trace's thinking, the forced answers (thinking closed, answer demanded at once) estimate the model's
belief b_t, and the rollouts (thinking continued from the cut) estimate its value V_t, the chance it ends correct.
All traces are kept: plan_execution reads them too.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from data.answers import boxed
from data.benchmarks import INSTRUCTION, load

RUNS = Path(__file__).resolve().parents[2] / "runs" / "belief_tracking"
CONTEXT = 8192
FORCE = "\n</think>\n\nThe final answer is $\\boxed{"


def thinking(tokenizer, text):
    """Token ids of the trace's thinking, up to the end-of-thinking tag."""
    end = text.find("</think>")
    return tokenizer(text[:end if end >= 0 else len(text)], add_special_tokens=False).input_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "math500"])
    parser.add_argument("--questions", type=int, default=150)
    parser.add_argument("--traces", type=int, default=16)
    parser.add_argument("--tracked", type=int, default=4)
    parser.add_argument("--cuts", type=int, default=8)
    parser.add_argument("--forced", type=int, default=16)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    llm = LLM(args.model, max_model_len=CONTEXT, gpu_memory_utilization=0.6)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sample = dict(temperature=0.6, top_p=0.95)
    out = RUNS / args.model.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        rows = load(name)
        rows = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:args.questions]]
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": f"{r['problem']}\n{INSTRUCTION}"}],
                                                 tokenize=False, add_generation_prompt=True) for r in rows]
        traces = llm.generate(prompts, SamplingParams(n=args.traces, max_tokens=args.max_tokens, **sample))
        records, cuts = [], []
        for row, prompt, output in zip(rows, prompts, traces):
            texts = [o.text for o in output.outputs]
            records.append({**row, "traces": [{"text": t, "answer": boxed(t)} for t in texts], "tracked": []})
            for j, text in enumerate(texts[:args.tracked]):
                ids = thinking(tokenizer, text)
                records[-1]["tracked"].append({"trace": j, "thinking_tokens": len(ids), "cuts": []})
                for position in np.linspace(0, len(ids), args.cuts).round().astype(int):
                    cuts.append((len(records) - 1, j, int(position), prompt + tokenizer.decode(ids[:position])))
        forced = llm.generate([c[3] + FORCE for c in cuts],
                              SamplingParams(n=args.forced, temperature=1.0, max_tokens=40))
        rollouts = llm.generate([c[3] for c in cuts],
                                [SamplingParams(n=args.rollouts, max_tokens=max(256, args.max_tokens - c[2]), **sample)
                                 for c in cuts])
        for (record, j, position, _), f, r in zip(cuts, forced, rollouts):
            tracked = next(t for t in records[record]["tracked"] if t["trace"] == j)
            tracked["cuts"].append({"position": position,
                                    "forced": [boxed("\\boxed{" + o.text) for o in f.outputs],
                                    "rollouts": [boxed(o.text) for o in r.outputs]})
        (out / f"{name}.json").write_text(json.dumps(records))
        print(f"{args.model} {name}: {len(records)} questions, {len(cuts)} cuts", flush=True)


if __name__ == "__main__":
    main()
