"""Evidence for composition: one stream per (agent, round), loaded from debate traces or zero-shot records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Stream:
    stream_id: str
    agent: int
    round: int
    model: str
    answer: Optional[str]
    samples: List[Optional[str]]
    tokens: Optional[int] = None
    signals: Dict[str, Optional[float]] = field(default_factory=dict)

    def first(self, budget: Optional[int]) -> List[Optional[str]]:
        """The first `budget` requested samples in stored order, invalid ones included."""
        return self.samples if budget is None else self.samples[:budget]


@dataclass
class Item:
    example_id: str
    question: str
    gold: str
    streams: List[Stream]

    def round_streams(self, rounds: Sequence[int]) -> List[Stream]:
        return [stream for stream in self.streams if stream.round in rounds]


def from_debate_traces(traces, rounds: Sequence[int] = (0,), sample_key: str = "consistency_t0.7") -> List[Item]:
    items = []
    for trace in traces:
        streams = []
        for turn in sorted(trace.turns, key=lambda t: (t.round, t.agent)):
            if turn.round not in rounds:
                continue
            streams.append(Stream(stream_id=turn.id, agent=turn.agent, round=turn.round, model=turn.model,
                                  answer=turn.answer, samples=turn.details.get("sampled_answers", {}).get(sample_key, []),
                                  tokens=turn.output_tokens or len(turn.logprobs or []),
                                  signals=dict(turn.confidence)))
        items.append(Item(trace.example_id, trace.question, trace.gold, streams))
    return items


def from_zero_shot(paths: Dict[str, Path], sample_key: str = "consistency_t0.7") -> List[Item]:
    """Zero-shot records from several models keyed by model name; every model answers the same examples."""
    per_model = {model: {row["id"]: row for row in _read(path)} for model, path in paths.items()}
    shared = set.intersection(*(set(rows) for rows in per_model.values()))
    items = []
    for example_id in sorted(shared, key=_sort_key):
        streams = []
        for agent, (model, rows) in enumerate(per_model.items()):
            row = rows[example_id]
            streams.append(Stream(stream_id=f"{example_id}:r0:a{agent}", agent=agent, round=0, model=model,
                                  answer=row["prediction"], samples=row.get("sampled_answers", {}).get(sample_key, []),
                                  tokens=len(row.get("token_logprobs", {}).get("response") or []) or None,
                                  signals=dict(row["confidence"])))
        first = next(iter(per_model.values()))[example_id]
        items.append(Item(example_id, first.get("question", ""), first["gold"], streams))
    return items


def _read(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _sort_key(example_id: str):
    tail = example_id.rsplit("-", 1)[-1]
    return (int(tail), example_id) if tail.isdigit() else (0, example_id)
