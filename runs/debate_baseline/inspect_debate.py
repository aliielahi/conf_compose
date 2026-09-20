"""Summarize a debate trace: accuracy per agent and round, answer flips, confidence, and label suspects."""

import argparse
from pathlib import Path

import numpy as np

from conf_compose.constants import TASKS
from conf_compose.data import get_task
from conf_compose.debate import flips, label_suspects, read_traces
from conf_compose.debate.analysis import is_correct
from conf_compose.utils.metrics import auroc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    return parser.parse_args()


def main():
    args = parse_args()
    task = get_task(args.task)
    traces = read_traces(Path(args.trace))
    suspects = set(label_suspects(task, traces))
    rounds = max(turn.round for turn in traces[0].turns)
    agents = sorted({turn.agent for turn in traces[0].turns})
    signals = sorted({name for turn in traces[0].turns for name in turn.confidence})
    print(f"{len(traces)} examples | {len(agents)} agents | {rounds + 1} rounds | signals: {', '.join(signals) or '-'}")
    if suspects:
        print(f"label suspects (all agents agree, gold differs): {len(suspects)} -> {sorted(suspects)}\n"
              f"  these stay in every metric; the bracketed column is only a sensitivity check and some of them\n"
              f"  are genuine confident collective failures, which is exactly what we study")

    for agent in agents:
        for round_index in range(rounds + 1):
            turns = [t for trace in traces for t in trace.round_turns(round_index) if t.agent == agent]
            correct = sum(is_correct(task, trace, t.answer) for trace in traces
                          for t in trace.round_turns(round_index) if t.agent == agent)
            model = turns[0].model if turns else "?"
            means = {name: np.mean([t.confidence[name] for t in turns if t.confidence.get(name) is not None])
                     for name in signals if any(t.confidence.get(name) is not None for t in turns)}
            summary = " ".join(f"{name}={value:.2f}" for name, value in means.items())
            print(f"agent{agent} round{round_index} {model:<16} acc={correct}/{len(turns)} {summary}")

    changes = [change for trace in traces for change in flips(task, trace)]
    helped = sum(c["now_correct"] and not c["was_correct"] for c in changes)
    hurt = sum(c["was_correct"] and not c["now_correct"] for c in changes)
    print(f"\nflips: {len(changes)} (helped {helped}, hurt {hurt})")
    for change in changes:
        print(f"  {change['example_id']} agent{change['agent']}: {change['before']} -> {change['after']} "
              f"({'ok' if change['was_correct'] else 'wrong'} -> {'ok' if change['now_correct'] else 'wrong'})")

    if signals:
        print("\nAUROC over all turns [sensitivity: same metric without label suspects]:")
        for name in signals:
            rows = [(t.confidence[name], is_correct(task, trace, t.answer), trace.example_id in suspects)
                    for trace in traces for t in trace.turns if t.confidence.get(name) is not None]
            values = np.array([r[0] for r in rows]); labels = np.array([r[1] for r in rows], float)
            keep = np.array([not r[2] for r in rows])
            clean = auroc(values[keep], labels[keep]) if 0 < labels[keep].sum() < keep.sum() else float("nan")
            print(f"  {name:<22} {auroc(values, labels):.3f} [{clean:.3f}]  n={len(rows)} errors={int((1 - labels).sum())}")


if __name__ == "__main__":
    main()
