"""Debate traces: one turn per agent per round, one JSONL line per example, resumable."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

Message = Dict[str, str]


@dataclass
class Turn:
    example_id: str
    agent: int
    model: str
    round: int
    parents: List[str]
    messages: List[Message]
    response: str
    answer: Optional[str]
    explicit_answer: bool
    finish_reason: Optional[str]
    logprobs: Optional[List[float]] = field(default=None, repr=False)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    seconds: Optional[float] = None
    execution: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)
    confidence: Dict[str, Optional[float]] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def id(self) -> str:
        return turn_id(self.example_id, self.round, self.agent)


@dataclass
class ExampleTrace:
    example_id: str
    question: str
    gold: str
    turns: List[Turn] = field(default_factory=list)

    def round_turns(self, round_index: int) -> List[Turn]:
        return [turn for turn in self.turns if turn.round == round_index]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def turn_id(example_id: str, round_index: int, agent: int) -> str:
    return f"{example_id}:r{round_index}:a{agent}"


def append_traces(path: Path, traces: Sequence[ExampleTrace]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write("".join(json.dumps(trace.to_dict()) + "\n" for trace in traces))


def read_traces(path: Path) -> List[ExampleTrace]:
    """Read saved traces, ignoring a truncated final line from an interrupted write."""
    if not path.exists():
        return []
    traces = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            warnings.warn(f"{path}: ignoring incomplete line {number}")
            continue
        turns = [Turn(**turn) for turn in payload.pop("turns")]
        traces.append(ExampleTrace(turns=turns, **payload))
    return traces


def completed_ids(path: Path) -> set:
    """Examples that finished cleanly; ones with a failed turn are retried."""
    return {trace.example_id for trace in read_traces(path)
            if all(turn.error is None and turn.response for turn in trace.turns)}
