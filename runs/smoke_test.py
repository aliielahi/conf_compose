"""Smoke test: load one model and exercise the shared generate/prompt interface."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.llm_calls import LLM


def section(title):
    print(f"\n=== {title} ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hf/q3-4bi")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    start = time.time()
    llm = LLM(args.model, max_tokens=args.max_tokens, cache_dir=args.cache_dir)
    print(f"loaded {llm} in {time.time() - start:.1f}s")

    section("single prompt")
    print(llm("Hi! Introduce yourself in one sentence."))

    section("batch with system prompt")
    replies = llm(
        ["What is 2 + 2?", "Name a primary color."],
        system="Answer with a single word.",
        max_tokens=8,
    )
    for reply in replies:
        print(repr(reply))

    section("multi-turn conversation")
    print(llm([
        {"role": "user", "content": "My name is Ali."},
        {"role": "assistant", "content": "Nice to meet you, Ali."},
        {"role": "user", "content": "What is my name?"},
    ]))

    section("generation with logprobs")
    g = llm.generate("Is the sky blue? Answer true or false.", max_tokens=4, logprobs=True)
    print(f"text={g.text!r} finish={g.finish_reason} tokens={g.tokens}")
    print(f"logprobs={g.logprobs} mean={g.mean_logprob}")

    section("self-consistency sampling (n=3, temperature=0.7)")
    for i, sample in enumerate(llm("Give me a random fruit name.", n=3, temperature=0.7, max_tokens=8)):
        print(i, repr(sample))

    print(f"\ndone in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
