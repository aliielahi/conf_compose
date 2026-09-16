"""P(True) self-verification: normalized probability that the model labels its own solution correct."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from prompts import VerificationPrompts


class SelfVerification:
    def __init__(self, llm, labels: Sequence[str] = VerificationPrompts.labels, system: Optional[str] = None):
        if not hasattr(llm, "score"):
            raise TypeError(f"{llm!r} cannot score continuations; use a local model")
        self.llm = llm
        self.labels = tuple(labels)
        self.system = system

    def estimate(self, task, examples, responses: Sequence[str]) -> List[float]:
        if len(examples) != len(responses):
            raise ValueError("examples and responses must have equal length")
        prompts = [VerificationPrompts.check(question=example.question, response=response)
                   for example, response in zip(examples, responses)]
        scores = {label: self.llm.score(prompts, [label] * len(prompts), system=self.system) for label in self.labels}
        true_label, false_label = self.labels
        return [_sigmoid(sum(scores[true_label][i]) - sum(scores[false_label][i])) for i in range(len(prompts))]


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
