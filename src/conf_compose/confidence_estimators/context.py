"""What every estimator scores: a claim, the response that made it, and the context that produced it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

Message = Dict[str, str]


@dataclass
class Target:
    example: Any
    conversation: List[Message]
    response: str
    answer: Optional[str]
    logprobs: Optional[List[float]] = field(default=None, repr=False)
    system: Optional[str] = None

    @property
    def prior_context(self) -> List[Message]:
        """The messages the agent saw before answering (its own response excluded)."""
        return list(self.conversation)

    @property
    def scored_conversation(self) -> List[Message]:
        """Context plus the response being scored, for follow-up questions about it."""
        return [*self.conversation, {"role": "assistant", "content": self.response}]


def zero_shot_targets(task, examples, generations) -> List[Target]:
    """Targets for single-turn answers: the prompt is the whole context."""
    targets = []
    for example, generation in zip(examples, generations, strict=True):
        answer = task.extract_answer(generation.text)
        targets.append(Target(example=example,
                              conversation=[{"role": "user", "content": task.prompt(example)}],
                              response=generation.text,
                              answer=answer.text if answer else None,
                              logprobs=generation.logprobs))
    return targets
