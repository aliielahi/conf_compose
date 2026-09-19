"""Generate debate traces for one task: N agents, R rounds, resumable, one JSONL per run."""

import argparse
import json
import time
from pathlib import Path

from conf_compose.constants import CACHE_DIR, DEBATE, RESULTS_DIR, TASKS, VLLM
from conf_compose.data import get_task
from conf_compose.debate import DebateConfig, run_debate
from conf_compose.utils.llm_calls import LLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEBATE["models"], help="one model per agent")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--n", type=int, help="examples; default from constants")
    parser.add_argument("--rounds", type=int, default=DEBATE["rounds"])
    parser.add_argument("--share-confidence", action="store_true", help="show peer confidence in later rounds")
    parser.add_argument("--execution", default=DEBATE["execution"], help="id separating fresh sampling runs")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--batch-size", type=int, default=100, help="examples traced before appending to disk")
    parser.add_argument("--gpu-memory-utilization", type=float, default=VLLM["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=VLLM["max_model_len"])
    parser.add_argument("--verbose", action="store_true", help="show vLLM engine logs")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / "debate"))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    return parser.parse_args()


def load_models(args):
    memory = args.gpu_memory_utilization / len(args.models)
    llms = []
    for model in args.models:
        llms.append(LLM(model, cache_dir=args.cache_dir, execution=args.execution, on_error="return",
                        gpu_memory_utilization=memory, max_model_len=args.max_model_len, quiet=not args.verbose,
                        engine_kwargs={"max_num_seqs": VLLM["max_num_seqs"],
                                       "max_num_batched_tokens": VLLM["max_num_batched_tokens"]}))
    return llms


def main():
    args = parse_args()
    defaults = TASKS[args.task]
    task = get_task(args.task)
    examples = task.load(args.split, n=args.n or defaults["n_test"])
    config = DebateConfig(models=tuple(args.models), rounds=args.rounds, share_confidence=args.share_confidence,
                          execution=args.execution, max_tokens=args.max_tokens or defaults["max_tokens"],
                          word_limit=defaults["word_limit"])

    start = time.time()
    llms = load_models(args)
    print(f"loaded {len(llms)} agents in {time.time() - start:.0f}s | {args.task}/{args.split} n={len(examples)} "
          f"rounds={config.rounds} execution={config.execution}")

    agents = "_".join(model.split("/")[-1] for model in args.models)
    name = f"{agents}_r{config.rounds}_{config.execution}_n{len(examples)}.jsonl"
    out_path = Path(args.out_dir) / args.task / name
    traces = run_debate(llms, task, examples, config, out_path, batch_size=args.batch_size)

    flips = sum(_flipped(trace) for trace in traces)
    print(f"\ntraced {len(traces)} examples in {time.time() - start:.0f}s | answer changed after debate: {flips}")
    (out_path.with_suffix(".config.json")).write_text(json.dumps({"args": vars(args), "config": config.to_dict()},
                                                                 indent=2))
    print(f"saved {out_path}")


def _flipped(trace) -> bool:
    first, last = trace.round_turns(0), trace.round_turns(max(turn.round for turn in trace.turns))
    return any(a.answer != b.answer for a, b in zip(first, last))


if __name__ == "__main__":
    main()
