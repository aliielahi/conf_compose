"""The judge combines a panel's evidence into one confidence in an answer it is given, not one it picks."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from conf_compose.confidence_estimators import SelfVerification, Target

from .prompts import HEADER, INSTRUCTION, MEMBER, SCORING, VIEWS

# The judge states its confidence in the same generation; the other two are probes over that verdict.
CONFIDENCE_MODES = ("verbalized", "ptrue", "seqprob")
# The judge may quote members' confidences while reasoning, so only the last labelled value is its own.
_RATING = re.compile(r"confidence\**\s*[:=]\s*\**\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_JUSTIFICATION = re.compile(r"justification\**\s*[:=]\s*\**\s*(.+)", re.IGNORECASE)


@dataclass
class PanelEntry:
    """One panel member as the judge may see it."""
    model: str
    answer: Optional[str]
    reasoning: str = ""
    confidence: Optional[float] = None


@dataclass
class PanelView:
    """One example: what the panel produced, and the aggregated answer the judge must rate."""
    example_id: str
    question: str
    gold: str
    entries: List[PanelEntry]
    final_answer: Optional[str]


@dataclass
class JudgeConfig:
    view: str = "reasoning_confidence"
    confidence_method: str = "consistency_t0.7"
    modes: Sequence[str] = ("verbalized",)
    word_limit: int = 150
    max_tokens: int = 512
    temperature: float = 0.0
    shuffle: bool = True
    context: Optional[int] = None   # the judge's window; a prompt that cannot fit is refused, never truncated

    def __post_init__(self):
        if self.view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}")
        for mode in self.modes:
            if mode not in CONFIDENCE_MODES:
                raise ValueError(f"confidence mode must be one of {CONFIDENCE_MODES}")


@dataclass
class Verdict:
    """The judge's confidence in the given answer, read three ways from one generation."""
    example_id: str
    final_answer: Optional[str]
    response: str
    prompt: str
    order: List[str]
    justification: str = ""
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    verbalized: Optional[float] = None
    ptrue: Optional[float] = None
    seqprob: Optional[float] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.example_id, "final_answer": self.final_answer,
                "justification": self.justification, "response": self.response,
                "finish_reason": self.finish_reason, "prompt_tokens": self.prompt_tokens,
                "output_tokens": self.output_tokens,
                "prompt": self.prompt, "order": self.order, "verbalized": self.verbalized,
                "ptrue": self.ptrue, "seqprob": self.seqprob, "candidates": self.candidates}


class Judge:
    """One generation per example; P(True) and candidate scoring are probes, not further generations."""

    def __init__(self, llm, task, config: Optional[JudgeConfig] = None):
        self.llm = llm
        self.task = task
        self.config = config or JudgeConfig()

    def run(self, views: Sequence[PanelView]) -> List[Verdict]:
        if not views:
            return []
        built = [self.build_prompt(view) for view in views]
        prompts = [prompt for prompt, _ in built]
        lengths = self.check_fit(views, prompts)
        generations = self.llm.generate(prompts, max_tokens=self.config.max_tokens,
                                        temperature=self.config.temperature)
        verdicts = [
            Verdict(view.example_id, view.final_answer, generation.text, prompt, list(order),
                    justification=_justification(generation.text),
                    finish_reason=generation.finish_reason,
                    prompt_tokens=generation.input_tokens or length,
                    output_tokens=generation.output_tokens,
                    verbalized=_rating(generation.text) if "verbalized" in self.config.modes else None)
            for view, (prompt, order), generation, length in zip(views, built, generations, lengths)
        ]
        if "ptrue" in self.config.modes:
            self._add_ptrue(views, verdicts)
        if "seqprob" in self.config.modes:
            self._add_seqprob(views, verdicts)
        return verdicts

    def check_fit(self, views: Sequence[PanelView], prompts: List[str]) -> List[Optional[int]]:
        """Count every prompt with the judge's own tokenizer and chat template; refuse the batch if one overflows."""
        if not self.config.context or not hasattr(self.llm, "encode"):
            return [None] * len(prompts)
        rendered = [self.llm.render(None, [{"role": "user", "content": prompt}]) for prompt in prompts]
        lengths = [len(ids) for ids in self.llm.encode(rendered)["input_ids"]]
        budget = self.config.context - self.config.max_tokens
        over = [(view.example_id, n) for view, n in zip(views, lengths) if n > budget]
        if over:
            raise ValueError(f"{len(over)}/{len(prompts)} prompt(s) exceed {budget} tokens "
                             f"(context {self.config.context} - max_tokens {self.config.max_tokens}); "
                             f"longest {max(n for _, n in over)} at {max(over, key=lambda o: o[1])[0]}")
        return lengths

    def build_prompt(self, view: PanelView) -> tuple:
        """Members are anonymised and their order shuffled per example, so position carries no information."""
        members, order = self._members(view)
        prompt = HEADER(question=view.question, members=members, final_answer=view.final_answer,
                        instruction=INSTRUCTION[self.config.view], word_limit=self.config.word_limit)
        return prompt, order

    def _members(self, view: PanelView) -> tuple:
        entries = list(view.entries)
        if self.config.shuffle:
            random.Random(view.example_id).shuffle(entries)
        template = MEMBER[self.config.view]
        blocks, order = [], []
        for index, entry in enumerate(entries):
            name = f"Model {chr(ord('A') + index)}"
            order.append(entry.model)
            blocks.append(template(name=name, answer=entry.answer, reasoning=entry.reasoning,
                                   method=self.config.confidence_method,
                                   confidence="unavailable" if entry.confidence is None
                                   else f"{entry.confidence:.2f}"))
        return "\n\n".join(blocks), order

    def _add_ptrue(self, views: Sequence[PanelView], verdicts: List[Verdict]) -> None:
        """Ask the judge, inside its own conversation, whether the final answer is correct: one token."""
        from conf_compose.data import Example

        scored = [(view, verdict) for view, verdict in zip(views, verdicts) if verdict.final_answer]
        if not scored:
            return
        targets = [Target(example=Example(view.example_id, view.question, view.gold),
                          conversation=[{"role": "user", "content": verdict.prompt}],
                          response=verdict.response, answer=verdict.final_answer)
                   for view, verdict in scored]
        estimator = SelfVerification(self.llm, context=True)
        for (_, verdict), value in zip(scored, estimator.estimate(self.task, targets)):
            verdict.ptrue = value

    def _add_seqprob(self, views: Sequence[PanelView], verdicts: List[Verdict]) -> None:
        """Teacher-force each panel answer after a plain context, then normalise to get P(final answer)."""
        requests, slots, pools, contexts = [], [], [], []
        for index, (view, verdict) in enumerate(zip(views, verdicts)):
            pool = self._candidates(view)
            pools.append(pool)
            members, _ = self._members(view)
            contexts.append(SCORING(question=view.question, members=members))
            for position, answer in enumerate(pool):
                requests.append((contexts[index], answer))
                slots.append((index, position))
        if not requests:
            return
        scores = self.llm.score([context for context, _ in requests],
                                [answer for _, answer in requests],
                                [self.task.answer_prefix] * len(requests))
        per_example: List[Dict[str, List[float]]] = [{} for _ in verdicts]
        for (index, position), row in zip(slots, scores):
            if row:
                per_example[index][pools[index][position]] = row

        for verdict, pool, scored in zip(verdicts, pools, per_example):
            verdict.candidates = [{"answer": answer, "logprobs": scored.get(answer)} for answer in pool]
            if not scored or verdict.final_answer is None:
                continue
            chosen = next((a for a in scored if self.task.equivalent(a, verdict.final_answer)), None)
            if chosen is None:
                continue
            totals = {answer: sum(row) for answer, row in scored.items()}
            top = max(totals.values())
            weights = {answer: math.exp(value - top) for answer, value in totals.items()}
            verdict.seqprob = weights[chosen] / sum(weights.values())

    def _candidates(self, view: PanelView) -> List[str]:
        """The panel's distinct answers, including the aggregated one, as the space to normalise over."""
        answers = [entry.answer for entry in view.entries if entry.answer is not None]
        if view.final_answer is not None:
            answers.append(view.final_answer)
        pool: List[str] = []
        for answer in answers:
            if not any(self.task.equivalent(existing, answer) for existing in pool):
                pool.append(answer)
        return pool


def _justification(text: str) -> str:
    """Saved only; nothing reads it yet. Falls back to the last line before the rating when unlabelled."""
    matches = _JUSTIFICATION.findall(text)
    if matches:
        return matches[-1].strip()
    ratings = list(_RATING.finditer(text))
    before = text[:ratings[-1].start()] if ratings else text
    lines = [line.strip() for line in before.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _rating(text: str, scale: float = 10) -> Optional[float]:
    matches = _RATING.findall(text)
    if not matches:
        return None
    value = float(matches[-1]) / scale
    return value if 0 <= value <= 1 else None
