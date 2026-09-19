"""Attach confidence signals to a saved debate trace; writes <trace>.conf.jsonl, resumable per example."""

import argparse
import json
import time
from pathlib import Path

from conf_compose.constants import CACHE_DIR, DEBATE, TASKS, VLLM
from conf_compose.data import get_task
from conf_compose.debate import append_traces, completed_ids, read_traces
from conf_compose.debate.confidence import add_confidence
from conf_compose.pipelines import ConfidenceConfig
from conf_compose.utils.llm_calls import LLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, help="path to the debate trace JSONL")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--models", nargs="+", help="one model per agent; default from the trace")
    parser.add_argument("--execution", default=DEBATE["execution"])
    parser.add_argument("--consistency-samples", type=int)
    parser.add_argument("--no-consistency", action="store_true", help="skip the resampling estimator")
    parser.add_argument("--batch-size", type=int, default=50, help="examples scored before appending to disk")
    parser.add_argument("--gpu-memory-utilization", type=float, default=VLLM["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=VLLM["max_model_len"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    return parser.parse_args()


def agent_models(traces, override):
    if override:
        return override
    first = traces[0]
    return [turn.model for turn in sorted(first.round_turns(0), key=lambda t: t.agent)]


def main():
    args = parse_args()
    trace_path = Path(args.trace)
    out_path = trace_path.with_suffix(".conf.jsonl")
    traces = read_traces(trace_path)
    done = completed_ids(out_path)
    pending = [trace for trace in traces if trace.example_id not in done]
    models = agent_models(traces, args.models)
    print(f"{len(pending)} examples pending of {len(traces)} | agents: {', '.join(models)}")
    if not pending:
        return

    task = get_task(args.task)
    defaults = TASKS[args.task]
    config = ConfidenceConfig(max_tokens=defaults["max_tokens"], verbalized=True, verification=True,
                              consistency_temperatures=() if args.no_consistency else ConfidenceConfig().
                              consistency_temperatures,
                              consistency_samples=args.consistency_samples or ConfidenceConfig().consistency_samples)

    memory = args.gpu_memory_utilization / len(models)
    llms = [LLM(model, cache_dir=args.cache_dir, execution=args.execution, on_error="return",
                gpu_memory_utilization=memory, max_model_len=args.max_model_len, quiet=not args.verbose)
            for model in models]

    start, timings = time.time(), {}
    for begin in range(0, len(pending), args.batch_size):
        batch = pending[begin:begin + args.batch_size]
        add_confidence(llms, task, batch, config, timings)
        append_traces(out_path, batch)
        print(f"[{time.strftime('%H:%M:%S')}] scored {begin + len(batch)}/{len(pending)}", flush=True)

    signals = sorted({name for trace in pending for turn in trace.turns for name in turn.confidence})
    print(f"\ndone in {time.time() - start:.0f}s | signals per turn: {', '.join(signals)}")
    out_path.with_suffix(".timings.json").write_text(json.dumps(timings, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
