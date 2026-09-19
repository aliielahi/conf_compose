"""Confidence for debate turns: each agent scores its own turns on the exact context it saw."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from conf_compose.confidence_estimators import Target
from conf_compose.data import Example
from conf_compose.pipelines.confidence import ConfidenceConfig, estimate_confidence

from .trace import ExampleTrace, Turn


def add_confidence(llms: Sequence, task, traces: Sequence[ExampleTrace],
                   config: Optional[ConfidenceConfig] = None,
                   timings: Optional[Dict[str, Dict[str, float]]] = None) -> List[ExampleTrace]:
    """Fill every turn's confidence dict in place, one estimator pass per agent."""
    config = config or ConfidenceConfig()
    for agent, llm in enumerate(llms):
        turns = [turn for trace in traces for turn in trace.turns if turn.agent == agent]
        targets = [_target(trace, turn) for trace in traces for turn in trace.turns if turn.agent == agent]
        agent_timings: Dict[str, Dict[str, float]] = {}
        for turn, output in zip(turns, estimate_confidence(llm, task, targets, config, agent_timings)):
            turn.confidence = output.signals
        if timings is not None:
            timings[f"agent{agent}"] = agent_timings
    return list(traces)


def _target(trace: ExampleTrace, turn: Turn) -> Target:
    example = Example(id=trace.example_id, question=trace.question, answer=trace.gold)
    return Target(example=example, conversation=turn.messages, response=turn.response, answer=turn.answer,
                  logprobs=turn.logprobs)


def final_answers(trace: ExampleTrace) -> List[Tuple[int, Optional[str]]]:
    """Each agent's answer after the last round, in agent order."""
    last = max(turn.round for turn in trace.turns)
    return [(turn.agent, turn.answer) for turn in sorted(trace.round_turns(last), key=lambda t: t.agent)]
