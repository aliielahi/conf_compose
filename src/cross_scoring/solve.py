"""Each scorer's own greedy final answer to every ProcessBench problem, the target of the belief signal."""

import argparse
import json
from pathlib import Path

from vllm import SamplingParams

from cross_scoring.teacher_forcing import load
from data.answers import boxed
from data.benchmarks import INSTRUCTION
from data.processbench import load as load_split

RUNS = Path(__file__).resolve().parents[2] / "runs" / "cross_scoring"
SPLITS = ("gsm8k", "math")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    args = parser.parse_args()
    out = RUNS / "answers" / args.model.replace("/", "__")
    if all((out / f"{split}.json").exists() for split in args.splits):
        return
    llm, tokenizer, limit = load(args.model)
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        problems = sorted({row["problem"] for row in load_split(split)})
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": f"{p}\n{INSTRUCTION}"}],
                                                 tokenize=False, add_generation_prompt=True) for p in problems]
        outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=min(4096, limit // 2)))
        (out / f"{split}.json").write_text(json.dumps({p: boxed(o.outputs[0].text) for p, o in zip(problems, outputs)}))
        print(f"{args.model} {split}: {len(problems)} problems solved", flush=True)


if __name__ == "__main__":
    main()
