"""A judge reads a panel of answers and returns one verdict, scored three ways from a single generation."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from conf_compose.confidence_estimators import SelfVerification, Target

from .prompts import HEADER, LEVELS, MEMBER

# The judge states a confidence in the same generation; the other two are read off the verdict afterwards.
CONFIDENCE_MODES = ("verbalized", "ptrue", "seqprob")
_RATING = re.compile(r"confidence\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass
class PanelEntry:
    """One panel member as the judge may see it."""
    model: str
    answer: Optional[str]
    reasoning: str = ""
    confidence: Optional[float] = None


@dataclass
class PanelView:
    """Everything about one example that the judge is given."""
    example_id: str
    question: str
    gold: str
    entries: List[PanelEntry]


@dataclass
class JudgeConfig:
    level: str = "answer_reasoning"
    shown_confidence: str = "consistency_t0.7"
    modes: Sequence[str] = CONFIDENCE_MODES
    word_limit: int = 150
    max_tokens: int = 512
    temperature: float = 0.0
    shuffle: bool = True

    def __post_init__(self):
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        for mode in self.modes:
            if mode not in CONFIDENCE_MODES:
                raise ValueError(f"confidence mode must be one of {CONFIDENCE_MODES}")


@dataclass
class Verdict:
    example_id: str
    answer: Optional[str]
    response: str
    prompt: str
    order: List[str]
    verbalized: Optional[float] = None
    ptrue: Optional[float] = None
    seqprob: Optional[float] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.example_id, "answer": self.answer, "response": self.response,
                "prompt": self.prompt, "order": self.order, "verbalized": self.verbalized,
                "ptrue": self.ptrue, "seqprob": self.seqprob, "candidates": self.candidates}


class Judge:
    """One generation per example; P(True) and candidate scoring are probes over that verdict, not generations."""

    def __init__(self, llm, task, config: Optional[JudgeConfig] = None):
        self.llm = llm
        self.task = task
        self.config = config or JudgeConfig()

    def run(self, views: Sequence[PanelView]) -> List[Verdict]:
        prompts, orders = zip(*(self.build_prompt(view) for view in views)) if views else ((), ())
        generations = self.llm.generate(list(prompts), max_tokens=self.config.max_tokens,
                                        temperature=self.config.temperature) if views else []
        verdicts = []
        for view, prompt, order, generation in zip(views, prompts, orders, generations):
            answer = self.task.extract_answer(generation.text)
            verdicts.append(Verdict(view.example_id, answer.text if answer else None, generation.text,
                                    prompt, list(order),
                                    verbalized=_rating(generation.text) if "verbalized" in self.config.modes
                                    else None))
        if "ptrue" in self.config.modes:
            self._add_ptrue(views, verdicts)
        if "seqprob" in self.config.modes:
            self._add_seqprob(views, verdicts)
        return verdicts

    def build_prompt(self, view: PanelView) -> tuple:
        """Members are anonymised and their order shuffled per example, so position carries no information."""
        entries = list(view.entries)
        if self.config.shuffle:
            random.Random(view.example_id).shuffle(entries)
        template = MEMBER[self.config.level]
        blocks, order = [], []
        for index, entry in enumerate(entries):
            name = f"Model {chr(ord('A') + index)}"
            order.append(entry.model)
            blocks.append(template(name=name, answer=entry.answer, reasoning=entry.reasoning,
                                   confidence="unavailable" if entry.confidence is None
                                   else f"{entry.confidence:.2f}"))
        prompt = HEADER(question=view.question, members="\n\n".join(blocks),
                        word_limit=self.config.word_limit, answer_prefix=self.task.answer_prefix)
        return prompt, order

    def _add_ptrue(self, views: Sequence[PanelView], verdicts: List[Verdict]) -> None:
        """Ask the judge whether its own verdict is correct, read from one next-token distribution."""
        from conf_compose.data import Example

        targets = [Target(example=Example(v.example_id, view.question, view.gold),
                          conversation=[{"role": "user", "content": v.prompt}],
                          response=v.response, answer=v.answer)
                   for view, v in zip(views, verdicts)]
        for verdict, value in zip(verdicts, SelfVerification(self.llm).estimate(self.task, targets)):
            verdict.ptrue = value

    def _add_seqprob(self, views: Sequence[PanelView], verdicts: List[Verdict]) -> None:
        """Teacher-force every candidate answer after the judge's own prompt, then normalise over them."""
        requests, slots = [], []
        pools = [self._candidates(view, verdict) for view, verdict in zip(views, verdicts)]
        for index, (verdict, pool) in enumerate(zip(verdicts, pools)):
            for position, answer in enumerate(pool):
                requests.append((verdict.prompt, self.task.answer_prefix, answer))
                slots.append((index, position))
        if not requests:
            return
        scores = self.llm.score([r[0] for r in requests], [r[2] for r in requests],
                                [r[1] for r in requests])
        logprobs: List[Dict[str, List[float]]] = [{} for _ in verdicts]
        for (index, position), row in zip(slots, scores):
            if row:
                logprobs[index][pools[index][position]] = row

        for verdict, pool, scored in zip(verdicts, pools, logprobs):
            verdict.candidates = [{"answer": answer, "logprobs": scored.get(answer)} for answer in pool]
            chosen = next((a for a in scored if self.task.equivalent(a, verdict.answer or "")), None)
            if chosen is None or not scored:
                continue
            totals = {answer: sum(row) for answer, row in scored.items()}
            top = max(totals.values())
            weights = {answer: math.exp(value - top) for answer, value in totals.items()}
            verdict.seqprob = weights[chosen] / sum(weights.values())

    def _candidates(self, view: PanelView, verdict: Verdict) -> List[str]:
        """The panel's distinct answers, plus the judge's own if it named something new."""
        answers = [entry.answer for entry in view.entries if entry.answer is not None]
        if verdict.answer is not None:
            answers.append(verdict.answer)
        pool: List[str] = []
        for answer in answers:
            if not any(self.task.equivalent(existing, answer) for existing in pool):
                pool.append(answer)
        return pool


def _rating(text: str, scale: float = 10) -> Optional[float]:
    match = _RATING.search(text)
    if match is None:
        return None
    value = float(match.group(1)) / scale
    return value if 0 <= value <= 1 else None
