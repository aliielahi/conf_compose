"""Candidate partitions and per-stream support: binary add-half for fixed events, categorical for vectors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .evidence import Item, Stream

OTHER = "__OTHER__"
EPSILON = 1e-3

# Full known answer space for closed-label tasks; open-answer tasks keep a residual OTHER state instead.
LABEL_SPACE = {"csqa": ("A", "B", "C", "D", "E"), "boolq": ("true", "false"), "prontoqa": ("true", "false"),
               "gpqa": ("A", "B", "C", "D")}


def label_space(task) -> Optional[Sequence[str]]:
    return LABEL_SPACE.get(getattr(task, "name", ""))


@dataclass
class Support:
    """Sample counts of one stream over a fixed partition; every lookup is resolved to its canonical class."""
    counts: Dict[str, int]
    valid: int
    requested: int
    candidates: Sequence[str]
    equivalent: Callable[[str, str], bool]

    @property
    def available(self) -> bool:
        return self.valid > 0

    def key(self, candidate: str) -> str:
        """Canonical class of a queried answer, so `12.0` finds the counts stored under `12`."""
        return next((c for c in self.candidates if self.equivalent(c, candidate)), OTHER)

    def count(self, candidate: str) -> int:
        return self.counts.get(self.key(candidate), 0)

    def raw(self, candidate: Optional[str]) -> Optional[float]:
        if candidate is None or not self.available:
            return None
        return self.count(candidate) / self.valid

    def binary(self, candidate: Optional[str], variant: str = "add_half") -> Optional[float]:
        """One fixed binary event, smoothed independently of how many other candidates happen to exist."""
        if candidate is None or not self.available:
            return None
        if variant == "epsilon":
            return min(max(self.count(candidate) / self.valid, EPSILON), 1 - EPSILON)
        return (self.count(candidate) + 0.5) / (self.valid + 1)

    def categorical(self, candidate: str, states: int, variant: str = "add_half") -> Optional[float]:
        """Uniform-Dirichlet smoothing over a state space fixed before these counts were seen."""
        if not self.available:
            return None
        if variant == "epsilon":
            return (self.count(candidate) / self.valid + EPSILON) / (1 + states * EPSILON)
        return (self.count(candidate) + 1 / states) / (self.valid + 1)


def candidate_set(task, item: Item, streams: Sequence[Stream]) -> List[str]:
    """Distinct valid main answers in round-then-agent order; the first occurrence represents the class."""
    candidates: List[str] = []
    for stream in sorted(streams, key=lambda s: (s.round, s.agent)):
        if stream.answer is None:
            continue
        if not any(task.equivalent(existing, stream.answer) for existing in candidates):
            candidates.append(stream.answer)
    return candidates


def state_space(task, candidates: Sequence[str], labels: Optional[Sequence[str]] = None) -> List[str]:
    """Probability states: this example's own options, else the task's label set, else candidates plus OTHER."""
    known = labels or label_space(task)
    return list(known) if known else [*candidates, OTHER]


def support_of(task, samples: Sequence[Optional[str]], candidates: Sequence[str]) -> Support:
    """Count samples into the candidate classes; anything else valid counts as OTHER."""
    counts: Dict[str, int] = {}
    valid = 0
    for sample in samples:
        if sample is None:
            continue
        valid += 1
        match = next((c for c in candidates if task.equivalent(c, sample)), OTHER)
        counts[match] = counts.get(match, 0) + 1
    return Support(counts, valid, len(samples), tuple(candidates), task.equivalent)


def support(task, stream: Stream, candidates: Sequence[str], budget: Optional[int] = None) -> Support:
    return support_of(task, stream.first(budget), candidates)


def logit(probability: float) -> float:
    probability = min(max(probability, EPSILON), 1 - EPSILON)
    return math.log(probability / (1 - probability))


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def softmax(scores: Dict[str, float]) -> Dict[str, float]:
    top = max(scores.values())
    weights = {key: math.exp(value - top) for key, value in scores.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}
