"""Sequence-probability confidence from token log-probs: mean, minimum and lowest-tail aggregations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from conf_compose.prompts import ContentFreeInputs

SCOPES = ("answer", "answer_no_reasoning", "response")


@dataclass
class SequenceProbResult:
    token_logprobs: List[float] = field(repr=False)
    tail_fraction: float
    null_mean_logprob: Optional[float] = None

    @property
    def n_tokens(self) -> int:
        return len(self.token_logprobs)

    @property
    def mean_logprob(self) -> float:
        return sum(self.token_logprobs) / self.n_tokens

    @property
    def min_logprob(self) -> float:
        return min(self.token_logprobs)

    @property
    def tail_logprob(self) -> float:
        k = max(1, math.ceil(self.tail_fraction * self.n_tokens))
        return sum(sorted(self.token_logprobs)[:k]) / k

    @property
    def confidence(self) -> float:
        return math.exp(self.mean_logprob)

    @property
    def min_confidence(self) -> float:
        return math.exp(self.min_logprob)

    @property
    def tail_confidence(self) -> float:
        return math.exp(self.tail_logprob)

    @property
    def debiased_confidence(self) -> Optional[float]:
        if self.null_mean_logprob is None:
            return None
        return _sigmoid(self.mean_logprob - self.null_mean_logprob)


def results_from_logprobs(token_logprobs: Sequence[Optional[Sequence[float]]],
                          tail_fraction: float = 0.1) -> List[Optional[SequenceProbResult]]:
    return [SequenceProbResult(list(lps), tail_fraction) if lps else None for lps in token_logprobs]


class SequenceProbability:
    def __init__(self, llm, scope: str = "response", debias: bool = True, tail_fraction: float = 0.1,
                 null_inputs: Sequence[str] = ContentFreeInputs.inputs, system: Optional[str] = None):
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")
        if not 0 < tail_fraction <= 1:
            raise ValueError("tail_fraction must be in (0, 1]")
        if not hasattr(llm, "score"):
            raise TypeError(f"{llm!r} cannot teacher-force continuations; use a local model")
        self.llm = llm
        self.scope = scope
        self.debias = debias
        self.tail_fraction = tail_fraction
        self.null_inputs = list(null_inputs)
        self.system = system

    def estimate(self, task, examples, responses: Sequence[str]) -> List[Optional[SequenceProbResult]]:
        pairs = zip(examples, responses, strict=True)
        rows = [self._spans(task, example, response) for example, response in pairs]
        valid = [i for i, row in enumerate(rows) if row is not None]
        prompts, prefixes, continuations = ([rows[i][k] for i in valid] for k in range(3))
        main = self.llm.score(prompts, continuations, prefixes, system=self.system)

        null_means: List[Optional[float]] = [None] * len(valid)
        if self.debias:
            null_prefix = "" if self.scope == "response" else task.answer_prefix
            per_null = [self.llm.score([task.null_prompt(null)] * len(valid), continuations,
                                       [null_prefix] * len(valid), system=self.system)
                        for null in self.null_inputs]
            null_means = [_logmeanexp([_mean(scores[j]) for scores in per_null if scores[j]])
                          for j in range(len(valid))]

        results: List[Optional[SequenceProbResult]] = [None] * len(rows)
        for j, i in enumerate(valid):
            if main[j]:
                results[i] = SequenceProbResult(main[j], self.tail_fraction, null_means[j])
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


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def _logmeanexp(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values) / len(values))
