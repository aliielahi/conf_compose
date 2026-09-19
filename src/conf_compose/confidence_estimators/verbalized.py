"""Verbalized confidence: ask the model to rate its own answer, repeated and averaged."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence

from conf_compose.prompts import VerbalizedPrompts

from .context import Target

_SCORE = re.compile(r"^\s*(?:confidence\s*[:=]?\s*)?(\d+(?:\.\d+)?)\s*(%|/\s*100|/\s*10)?\s*\.?\s*(?:\n|$)",
                    re.IGNORECASE)


@dataclass
class VerbalizedResult:
    confidence: Optional[float]
    scores: List[Optional[float]]
    raw: List[str]

    @property
    def n_failed(self) -> int:
        return sum(s is None for s in self.scores)


def parse_confidence(text: str, scale: float = 10) -> Optional[float]:
    match = _SCORE.match(text)
    if match is None:
        return None
    value, unit = float(match.group(1)), (match.group(2) or "").replace(" ", "")
    value /= 100 if unit in ("%", "/100") else scale
    return value if 0 <= value <= 1 else None


class VerbalizedConfidence:
    def __init__(self, llm, prompt=VerbalizedPrompts.rate, repeats: int = 3,
                 temperature: float = 0.3, top_p: float = 1.0, top_k: int = 0, max_tokens: int = 8,
                 scale: float = 10, system: Optional[str] = None):
        self.llm = llm
        self.prompt = prompt
        self.repeats = repeats
        self.sampling = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_tokens": max_tokens}
        self.scale = scale
        self.system = system

    def estimate(self, task, targets: Sequence[Target]) -> List[VerbalizedResult]:
        scored = [i for i, target in enumerate(targets) if target.answer is not None]
        conversations = [[*targets[i].scored_conversation,
                          {"role": "user", "content": self.prompt(answer=targets[i].answer)}] for i in scored]
        samples = self.llm.prompt(conversations, n=self.repeats, system=self.system,
                                  **self.sampling) if conversations else []
        results: List[VerbalizedResult] = [VerbalizedResult(None, [], []) for _ in targets]
        for i, raw in zip(scored, samples):
            raw = raw if isinstance(raw, list) else [raw]
            scores = [parse_confidence(text, self.scale) for text in raw]
            parsed = [s for s in scores if s is not None]
            results[i] = VerbalizedResult(statistics.mean(parsed) if parsed else None, scores, raw)
        return results
