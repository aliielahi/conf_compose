"""Consistency confidence: how often resampled responses reach the same answer as the main prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class ConsistencyResult:
    confidence: Optional[float]
    majority_answer: Optional[str]
    majority_fraction: float
    answers: List[Optional[str]]


class ConsistencyConfidence:
    def __init__(self, llm, samples: int = 5, temperature: float = 0.7, max_tokens: int = 512,
                 system: Optional[str] = None):
        self.llm = llm
        self.samples = samples
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system = system

    def estimate(self, task, examples, predictions: Sequence[Optional[str]]) -> List[ConsistencyResult]:
        prompts = [task.prompt(example) for example in examples]
        sampled = self.llm.prompt(prompts, n=self.samples, temperature=self.temperature,
                                  max_tokens=self.max_tokens, system=self.system)
        results = []
        for prediction, responses in zip(predictions, sampled):
            responses = responses if isinstance(responses, list) else [responses]
            answers = [answer.text if (answer := task.extract_answer(r)) else None for r in responses]
            clusters = _cluster(task, [a for a in answers if a is not None])
            majority, majority_count = max(clusters, key=lambda c: c[1], default=(None, 0))
            agreement = None if prediction is None else sum(
                a is not None and task.equivalent(a, prediction) for a in answers) / len(answers)
            results.append(ConsistencyResult(agreement, majority, majority_count / len(answers), answers))
        return results


def _cluster(task, answers: Sequence[str]) -> List[List]:
    clusters: List[List] = []
    for answer in answers:
        match = next((c for c in clusters if task.equivalent(c[0], answer)), None)
        if match is None:
            clusters.append([answer, 1])
        else:
            match[1] += 1
    return clusters
