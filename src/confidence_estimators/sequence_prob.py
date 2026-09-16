"""Sequence-probability confidence: exp(mean token log-prob), optionally debiased against content-free inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

from prompts import ContentFreeInputs

SCOPES = ("answer", "answer_no_reasoning", "response")


@dataclass
class SequenceProbResult:
    confidence: float
    mean_logprob: float
    n_tokens: int
    null_mean_logprob: Optional[float] = None
    debiased_confidence: Optional[float] = None


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def _logmeanexp(values: Sequence[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values) / len(values))


class SequenceProbability:
    def __init__(self, llm, scope: str = "answer", debias: bool = True,
                 null_inputs: Sequence[str] = ContentFreeInputs.inputs, system: Optional[str] = None):
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")
        if not hasattr(llm, "score"):
            raise TypeError(f"{llm!r} cannot teacher-force continuations; use a local HF model")
        self.llm = llm
        self.scope = scope
        self.debias = debias
        self.null_inputs = list(null_inputs)
        self.system = system

    def estimate(self, task, examples, responses: Sequence[str]) -> List[Optional[SequenceProbResult]]:
        rows = [self._spans(task, ex, resp) for ex, resp in zip(examples, responses)]
        valid = [i for i, row in enumerate(rows) if row is not None]
        prompts = [rows[i][0] for i in valid]
        prefixes = [rows[i][1] for i in valid]
        continuations = [rows[i][2] for i in valid]
        main = self.llm.score(prompts, continuations, prefixes, system=self.system)

        null_means: List[Optional[float]] = [None] * len(valid)
        if self.debias:
            null_prefix = "" if self.scope == "response" else task.answer_prefix
            per_null = [
                self.llm.score([task.null_prompt(null)] * len(valid), continuations,
                               [null_prefix] * len(valid), system=self.system)
                for null in self.null_inputs
            ]
            null_means = [_logmeanexp([_mean(scores[j]) for scores in per_null]) for j in range(len(valid))]

        results: List[Optional[SequenceProbResult]] = [None] * len(rows)
        for j, i in enumerate(valid):
            if not main[j]:
                continue
            mean = _mean(main[j])
            result = SequenceProbResult(confidence=math.exp(mean), mean_logprob=mean, n_tokens=len(main[j]))
            if null_means[j] is not None:
                result.null_mean_logprob = null_means[j]
                result.debiased_confidence = _sigmoid(mean - null_means[j])
            results[i] = result
        return results

    def _spans(self, task, example, response: str):
        prompt = task.prompt(example)
        if self.scope == "response":
            return (prompt, "", response) if response else None
        answer = task.extract_answer(response)
        if answer is None:
            return None
        if self.scope == "answer":
            return prompt, response[:answer.start], answer.text
        return prompt, task.answer_prefix, answer.text


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)
