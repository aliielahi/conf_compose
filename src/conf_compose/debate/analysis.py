"""Small checks on saved debate traces: answer flips and examples whose gold label looks wrong."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from conf_compose.data import Example

from .trace import ExampleTrace


def is_correct(task, trace: ExampleTrace, answer) -> bool:
    return task.is_correct(answer, Example(trace.example_id, trace.question, trace.gold))


def changed(task, before, after) -> bool:
    """True when two answers differ as answers, not merely as strings ("12" vs "$12" are the same)."""
    if before is None or after is None:
        return before != after
    return not task.equivalent(before, after)


def flips(task, trace: ExampleTrace) -> List[Dict[str, Any]]:
    """Answer changes between the first and last round, per agent."""
    last = max(turn.round for turn in trace.turns)
    changes = []
    for first_turn in sorted(trace.round_turns(0), key=lambda t: t.agent):
        final = next(t for t in trace.round_turns(last) if t.agent == first_turn.agent)
        if changed(task, first_turn.answer, final.answer):
            changes.append({"example_id": trace.example_id, "agent": first_turn.agent,
                            "before": first_turn.answer, "after": final.answer,
                            "was_correct": is_correct(task, trace, first_turn.answer),
                            "now_correct": is_correct(task, trace, final.answer)})
    return changes


def label_suspects(task, traces: Sequence[ExampleTrace]) -> List[str]:
    """Examples where every agent in every round gives the same answer, yet the gold label disagrees."""
    suspects = []
    for trace in traces:
        answers = [turn.answer for turn in trace.turns]
        if not answers or any(answer is None for answer in answers):
            continue
        unanimous = all(task.equivalent(answer, answers[0]) for answer in answers)
        if unanimous and not is_correct(task, trace, answers[0]):
            suspects.append(trace.example_id)
    return suspects
