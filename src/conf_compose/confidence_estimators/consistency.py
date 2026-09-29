"""Consistency confidence: agreement, margin and entropy of answers across resampled responses."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .context import Target


@dataclass
class ConsistencyResult:
    answers: List[Optional[str]]
    cluster_sizes: List[int]
    agreement: Optional[float]
    responses: List[str] = field(default_factory=list, repr=False)
    logprobs: List[Optional[List[float]]] = field(default_factory=list, repr=False)

    @property
    def samples(self) -> int:
        return len(self.answers)

    @property
    def majority_fraction(self) -> float:
        return max(self.cluster_sizes, default=0) / self.samples

    @property
    def margin(self) -> float:
        top = sorted(self.cluster_sizes, reverse=True) + [0, 0]
        return (top[0] - top[1]) / self.samples

    @property
    def valid_fraction(self) -> float:
        return sum(self.cluster_sizes) / self.samples

    @property
    def entropy_confidence(self) -> float:
        invalid = self.samples - sum(self.cluster_sizes)
        counts = [c for c in self.cluster_sizes + [1] * invalid if c]
        entropy = -sum(c / self.samples * math.log(c / self.samples) for c in counts)
        return 1 - entropy / math.log(self.samples) if self.samples > 1 else 1.0


class ConsistencyConfidence:
    def __init__(self, llm, samples: int = 10, temperature: float = 0.7, top_p: float = 1.0, top_k: int = 0,
                 max_tokens: int = 512, logprobs: bool = False, system: Optional[str] = None):
        """logprobs=True keeps each sample's per-token scores; it changes the request, so it re-samples."""
        self.llm = llm
        self.samples = samples
        self.sampling = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_tokens": max_tokens}
        self.logprobs = logprobs
        self.system = system

    def estimate(self, task, targets: Sequence[Target]) -> List[ConsistencyResult]:
        """Resample each agent's own incoming context and measure agreement with its target answer."""
        contexts = [target.prior_context for target in targets]
        sampled, scores = self._draw(contexts)
        results = []
        for target, responses, row in zip(targets, sampled, scores):
            prediction = target.answer
            answers = [answer.text if (answer := task.extract_answer(r)) else None for r in responses]
            valid = [a for a in answers if a is not None]
            agreement = None
            if prediction is not None:
                agreement = sum(task.equivalent(a, prediction) for a in valid) / len(answers)
            results.append(ConsistencyResult(answers, _cluster_sizes(task, valid), agreement,
                                             list(responses), row))
        return results

    def _draw(self, contexts):
        """Sampled texts and, when asked, the per-token log-probs that came back with them."""
        if not self.logprobs:
            sampled = self.llm.prompt(contexts, n=self.samples, system=self.system, **self.sampling)
            rows = [_as_list(r) for r in sampled]
            return rows, [[None] * len(row) for row in rows]
        generated = self.llm.generate(contexts, n=self.samples, system=self.system, logprobs=True,
                                      **self.sampling)
        rows = [_as_list(r) for r in generated]
        return ([[g.text for g in row] for row in rows],
                [[g.logprobs for g in row] for row in rows])


def _as_list(value):
    return value if isinstance(value, list) else [value]


def _cluster_sizes(task, answers: Sequence[str]) -> List[int]:
    representatives: List[str] = []
    sizes: List[int] = []
    for answer in answers:
        index = next((i for i, rep in enumerate(representatives) if task.equivalent(rep, answer)), None)
        if index is None:
            representatives.append(answer)
            sizes.append(1)
        else:
            sizes[index] += 1
    return sizes
