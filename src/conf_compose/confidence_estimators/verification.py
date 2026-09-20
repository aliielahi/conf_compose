"""P(True) self-verification from one next-token distribution: P(True) / (P(True) + P(False))."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from conf_compose.prompts import VerificationPrompts

from .context import Target


class SelfVerification:
    def __init__(self, llm, labels: Sequence[str] = VerificationPrompts.labels, top_logprobs: int = 20,
                 context: bool = False, claim: bool = False, system: Optional[str] = None):
        """context=True asks inside the agent's history; claim=True rates an answer with no response attached."""
        self.llm = llm
        self.context = context
        self.claim = claim
        self.true_label, self.false_label = labels
        self.top_logprobs = top_logprobs
        self.system = system

    def estimate(self, task, targets: Sequence[Target]) -> List[Optional[float]]:
        scored = [i for i, target in enumerate(targets) if target.answer is not None]
        prompts = [self._ask(targets[i]) for i in scored]
        generations = self.llm.generate(prompts, max_tokens=1, temperature=0.0, top_logprobs=self.top_logprobs,
                                        system=self.system) if prompts else []
        values: List[Optional[float]] = [None] * len(targets)
        self.missing_labels = 0
        for i, generation in zip(scored, generations):
            values[i] = self._probability(generation.top_logprobs[0]) if generation.top_logprobs else None
            self.missing_labels += values[i] is None
        return values

    def _ask(self, target: Target):
        if self.claim:
            return VerificationPrompts.check_claim(question=target.example.question, answer=target.answer)
        if not self.context:
            return VerificationPrompts.check(question=target.example.question, response=target.response,
                                             answer=target.answer)
        return [*target.scored_conversation,
                {"role": "user", "content": VerificationPrompts.check_own(answer=target.answer)}]

    def _probability(self, top: Dict[str, float]) -> Optional[float]:
        """None when a label is outside the returned tokens: unavailable, not a low probability."""
        true = _label_logprob(top, self.true_label)
        false = _label_logprob(top, self.false_label)
        return None if true is None or false is None else _sigmoid(true - false)


def _label_logprob(top: Dict[str, float], label: str) -> Optional[float]:
    matches = [logprob for token, logprob in top.items() if token.strip().lower() == label.lower()]
    return max(matches) if matches else None


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
