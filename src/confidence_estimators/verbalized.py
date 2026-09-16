"""Verbalized confidence: ask the model to rate its own answer, repeated and averaged."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence

DEFAULT_PROMPT = (
    "Rate your confidence in your final answer on a scale from 0 to 10, where 0 means you are guessing "
    "randomly, 5 means you are somewhat confident but unsure, and 10 means you are almost certain it is "
    "correct. Only output one integer in the [0, 10] range."
)

_SCORE = re.compile(r"(\d+(?:\.\d+)?)\s*(%|/\s*(?:10|100)\b)?")


@dataclass
class VerbalizedResult:
    confidence: Optional[float]
    scores: List[Optional[float]]
    raw: List[str]

    @property
    def n_failed(self) -> int:
        return sum(s is None for s in self.scores)


def parse_confidence(text: str, scale: float = 10) -> Optional[float]:
    match = _SCORE.search(text)
    if match is None:
        return None
    value, unit = float(match.group(1)), (match.group(2) or "").replace(" ", "")
    if unit in ("%", "/100"):
        value /= 100
    elif unit == "/10" or value <= scale:
        value /= scale
    else:
        return None
    return value if 0 <= value <= 1 else None


class VerbalizedConfidence:
    def __init__(self, llm, prompt: str = DEFAULT_PROMPT, repeats: int = 3, temperature: float = 0.3,
                 max_tokens: int = 8, scale: float = 10, system: Optional[str] = None):
        self.llm = llm
        self.prompt = prompt
        self.repeats = repeats
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.scale = scale
        self.system = system

    def estimate(self, prompts: Sequence[str], responses: Sequence[str]) -> List[VerbalizedResult]:
        conversations = [
            [{"role": "user", "content": p}, {"role": "assistant", "content": r},
             {"role": "user", "content": self.prompt}]
            for p, r in zip(prompts, responses)
        ]
        samples = self.llm.prompt(conversations, n=self.repeats, temperature=self.temperature,
                                  max_tokens=self.max_tokens, system=self.system)
        results = []
        for raw in samples:
            raw = raw if isinstance(raw, list) else [raw]
            scores = [parse_confidence(text, self.scale) for text in raw]
            parsed = [s for s in scores if s is not None]
            results.append(VerbalizedResult(statistics.mean(parsed) if parsed else None, scores, raw))
        return results
