"""P(True) self-verification from one next-token distribution: P(True) / (P(True) + P(False))."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from conf_compose.prompts import VerificationPrompts

from .context import Target


class SelfVerification:
    def __init__(self, llm, labels: Sequence[str] = VerificationPrompts.labels, top_logprobs: int = 20,
                 system: Optional[str] = None):
        self.llm = llm
        self.true_label, self.false_label = labels
        self.top_logprobs = top_logprobs
        self.system = system

    def estimate(self, task, targets: Sequence[Target]) -> List[Optional[float]]:
        scored = [i for i, target in enumerate(targets) if target.answer is not None]
        prompts = [VerificationPrompts.check(question=targets[i].example.question, response=targets[i].response,
                                             answer=targets[i].answer) for i in scored]
        generations = self.llm.generate(prompts, max_tokens=1, temperature=0.0, top_logprobs=self.top_logprobs,
                                        system=self.system) if prompts else []
        values: List[Optional[float]] = [None] * len(targets)
        for i, generation in zip(scored, generations):
            values[i] = self._probability(generation.top_logprobs[0]) if generation.top_logprobs else None
        return values

    def _probability(self, top: Dict[str, float]) -> float:
        floor = min(top.values()) - math.log(2)
        true = _label_logprob(top, self.true_label, floor)
        false = _label_logprob(top, self.false_label, floor)
        return _sigmoid(true - false)


def _label_logprob(top: Dict[str, float], label: str, floor: float) -> float:
    matches = [logprob for token, logprob in top.items() if token.strip().lower() == label.lower()]
    return max(matches) if matches else floor


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
