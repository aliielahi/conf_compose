"""Debate protocol settings: who debates, for how many rounds, and what peers get to see."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

from conf_compose.constants import DEBATE


@dataclass
class DebateConfig:
    models: Tuple[str, ...] = tuple(DEBATE["models"])
    rounds: int = DEBATE["rounds"]
    share_confidence: bool = DEBATE["share_confidence"]
    execution: str = DEBATE["execution"]
    max_tokens: int = 1024
    word_limit: int = 250
    temperature: float = 0.0
    sampling: Dict[str, Any] = field(default_factory=dict)

    @property
    def agents(self) -> int:
        return len(self.models)

    def generation_settings(self) -> Dict[str, Any]:
        return {"max_tokens": self.max_tokens, "temperature": self.temperature, "logprobs": True, **self.sampling}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
