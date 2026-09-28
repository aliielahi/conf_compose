"""Sequence-probability confidence from token log-probs: mean, minimum and lowest-tail aggregations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from conf_compose.prompts import ContentFreeInputs

from .context import Target

SCOPES = ("answer", "answer_no_reasoning", "response")

# Contexts a candidate can be scored in; `reasoned` fixes the boundary at the model's own answer start.
CONTEXTS = ("direct", "reasoned")


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

    def estimate(self, task, targets: Sequence[Target]) -> List[Optional[SequenceProbResult]]:
        rows = [self._spans(task, target) for target in targets]
        valid = [i for i, row in enumerate(rows) if row is not None]
        if not valid:
            return [None] * len(rows)
        prompts, prefixes, continuations = ([rows[i][k] for i in valid] for k in range(3))
        main = self.llm.score(prompts, continuations, prefixes, system=self.system)

        null_means: List[Optional[float]] = [None] * len(valid)
        if self.debias:
            null_prefix = "" if self.scope == "response" else task.answer_prefix
            per_null = [self.llm.score([task.null_prompt(null)] * len(valid), continuations,
                                       [null_prefix] * len(valid), system=self.system)
                        for null in self.null_inputs]  # content-free baseline stays single-turn by design
            null_means = [_logmeanexp([_mean(scores[j]) for scores in per_null if scores[j]])
                          for j in range(len(valid))]

        results: List[Optional[SequenceProbResult]] = [None] * len(rows)
        for j, i in enumerate(valid):
            if main[j]:
                results[i] = SequenceProbResult(main[j], self.tail_fraction, null_means[j])
        return results

    def _spans(self, task, target: Target):
        """response: scores the response itself; answer scopes: score target.answer, the designated claim."""
        context, response = target.conversation, target.response
        if self.scope == "response":
            return (context, "", response) if response else None
        if target.answer is None:
            return None
        if self.scope == "answer_no_reasoning":
            return context, task.answer_prefix, target.answer
        start = response.rfind(target.answer)
        return (context, response[:start], target.answer) if start >= 0 else None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def _logmeanexp(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values) / len(values))


@dataclass
class Candidate:
    """One answer string a model is asked to score, with enough identity to never lose what it was."""
    answer: str
    text: str
    option: Optional[str] = None
    source: str = "generated"

    def to_dict(self) -> dict:
        return {"answer": self.answer, "text": self.text, "option": self.option, "source": self.source}


class CandidateScorer:
    """Score any candidate answer under one model, whether or not that model ever produced it.

    Raw per-token log-probabilities are returned untouched; every derived score (mean, sum, minimum,
    bottom tail, debiasing, normalisation over a candidate set) is a downstream choice.
    """

    def __init__(self, llm, contexts: Sequence[str] = CONTEXTS,
                 nulls: Sequence[str] = ContentFreeInputs.inputs, debias: bool = True,
                 system: Optional[str] = None):
        for context in contexts:
            if context not in CONTEXTS:
                raise ValueError(f"context must be one of {CONTEXTS}")
        if not hasattr(llm, "score"):
            raise TypeError(f"{llm!r} cannot teacher-force continuations; use a local model")
        self.llm = llm
        self.contexts = tuple(contexts)
        self.nulls = list(nulls) if debias else []
        self.system = system

    def score(self, task, examples: Sequence[Any], responses: Sequence[Optional[str]],
              starts: Sequence[Optional[int]], candidates: Sequence[Sequence[Candidate]]) -> List[dict]:
        """One entry per example holding every candidate's token log-probs in every requested context."""
        out = [{"version": 1, "answer_prefix": task.answer_prefix, "answer_start": start,
                "candidates": [candidate.to_dict() for candidate in row]}
               for start, row in zip(starts, candidates)]

        for context in self.contexts:
            requests, slots = [], []
            for index, (example, response, start, row) in enumerate(
                    zip(examples, responses, starts, candidates)):
                prefix = self._prefix(task, response, start, context)
                for position, candidate in enumerate(row):
                    if prefix is None:
                        out[index]["candidates"][position][context] = None
                        out[index]["candidates"][position].setdefault("errors", {})[context] = "no answer span"
                        continue
                    requests.append((task.prompt(example), prefix, candidate.text))
                    slots.append((index, position))
            for (index, position), logprobs in zip(slots, self._run(requests)):
                out[index]["candidates"][position][context] = {"logprobs": logprobs} if logprobs else None

        for null in self.nulls:
            requests, slots = [], []
            for index, row in enumerate(candidates):
                for position, candidate in enumerate(row):
                    requests.append((task.null_prompt(null), task.answer_prefix, candidate.text))
                    slots.append((index, position))
            for (index, position), logprobs in zip(slots, self._run(requests)):
                entry = out[index]["candidates"][position].setdefault("null", {})
                entry[null] = logprobs or None
        return out

    def _prefix(self, task, response: Optional[str], start: Optional[int], context: str) -> Optional[str]:
        """`direct` asks the question alone; `reasoned` replays the model's own words up to its answer."""
        if context == "direct":
            return task.answer_prefix
        if not response or start is None:
            return None
        return response[:start]

    def _run(self, requests: Sequence[tuple]) -> List[List[float]]:
        if not requests:
            return []
        prompts, prefixes, continuations = ([r[k] for r in requests] for k in range(3))
        return self.llm.score(prompts, continuations, prefixes, system=self.system)
