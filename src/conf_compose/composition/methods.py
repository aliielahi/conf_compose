"""Composition methods: confidence for a fixed answer (family A), selection (family B), panel pooling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .candidates import (OTHER, Support, candidate_set, logit, sigmoid, softmax, state_space, support)
from .evidence import Item, Stream


@dataclass
class Prediction:
    """One method's output for one example: a score, and for family B the answer it selected."""
    score: Optional[float]
    is_probability: bool = True
    answer: Optional[str] = None
    logit: Optional[float] = None
    details: Dict[str, float] = field(default_factory=dict)


def pool_methods(scores: Sequence[float], prior: Optional[float] = None) -> Dict[str, Prediction]:
    """Fixed pooling rules over ratings of one shared target; logits are kept for calibration."""
    if not scores:
        return {}
    logits = [logit(value) for value in scores]
    total, mean = sum(logits), sum(logits) / len(logits)
    methods = {"mean": Prediction(sum(scores) / len(scores)),
               "logodds_sum": Prediction(sigmoid(total), logit=total),
               "logodds_mean": Prediction(sigmoid(mean), logit=mean)}
    if prior is not None:
        methods["prior_logodds_sum"] = Prediction(prior_corrected(logits, prior, 1.0))
        methods["prior_logodds_mean"] = Prediction(prior_corrected(logits, prior, 1 / len(logits)))
    return methods


def fixed_answer_methods(task, item: Item, streams: Sequence[Stream], target: Optional[str],
                         budget: Optional[int] = None, variant: str = "add_half",
                         target_tokens: Optional[int] = None,
                         prior: Optional[float] = None) -> Dict[str, Prediction]:
    """Confidence in one fixed answer; every method scores the same target and never changes it."""
    if target is None:
        return {}
    candidates = candidate_set(task, item, streams)
    if not any(task.equivalent(candidate, target) for candidate in candidates):
        candidates = [target, *candidates]
    supports = {stream.stream_id: support(task, stream, candidates, budget) for stream in streams}
    available = [s for s in streams if supports[s.stream_id].available]

    methods: Dict[str, Prediction] = {"constant_half": Prediction(0.5)}
    for stream in streams:
        methods[f"support_r{stream.round}_a{stream.agent}"] = Prediction(supports[stream.stream_id].raw(target))
    methods["length"] = Prediction(-math.log1p(target_tokens) if target_tokens else None, is_probability=False)

    votes = [s for s in streams if s.answer is not None]
    agreeing = [s for s in votes if task.equivalent(s.answer, target)]
    methods["vote_share"] = Prediction(len(agreeing) / len(votes) if votes else None)

    if available:
        smooth = [supports[s.stream_id].binary(target, variant) for s in available]
        methods.update(pool_methods(smooth, prior))
        methods["mean_support_raw"] = Prediction(sum(supports[s.stream_id].raw(target) for s in available) / len(available))
        if prior is not None:
            methods["prior_constant"] = Prediction(prior)
    return methods


def selection_methods(task, item: Item, streams: Sequence[Stream], budget: Optional[int] = None,
                      variant: str = "add_half") -> Dict[str, Prediction]:
    """Methods that pick the answer as well as scoring it; selection stays within the proposed answers."""
    candidates = candidate_set(task, item, streams)
    if not candidates:
        return {}
    states = state_space(task, candidates, item.labels)
    supports = {stream.stream_id: support(task, stream, candidates, budget) for stream in streams}
    available = [s for s in streams if supports[s.stream_id].available]
    vectors = {s.stream_id: {c: supports[s.stream_id].categorical(c, len(states), variant) for c in states}
               for s in available}

    methods: Dict[str, Prediction] = {}
    votes = [s for s in streams if s.answer is not None]
    if votes:
        counts = {c: sum(task.equivalent(s.answer, c) for s in votes) for c in candidates}
        winner = max(candidates, key=lambda c: (counts[c], -candidates.index(c)))
        methods["majority_vote"] = Prediction(counts[winner] / len(votes), answer=winner)

    if available:
        linear = {c: sum(vectors[s.stream_id][c] for s in available) / len(available) for c in states}
        methods["linear_pool"] = _select(candidates, linear)
        log_scores = {c: sum(math.log(vectors[s.stream_id][c]) for s in available) for c in states}
        methods["log_pool"] = _select(candidates, softmax(log_scores))
        methods["log_pool_mean"] = _select(candidates, softmax({c: v / len(available) for c, v in log_scores.items()}))
        best = max(available, key=lambda s: (supports[s.stream_id].raw(s.answer) or -1, -s.round, -s.agent))
        if best.answer is not None:
            methods["max_own_support"] = Prediction(supports[best.stream_id].raw(best.answer), answer=best.answer)
    return methods


def prior_corrected(logits: Sequence[float], prior: float, w: float) -> float:
    """sigma[w * sum_j logit(q_j) - (w*S - 1) * logit(pi)]; the prior cancels exactly at w = 1/S."""
    return sigmoid(w * sum(logits) - (w * len(logits) - 1) * logit(prior))


def majority_answer(task, streams: Sequence[Stream]) -> Optional[str]:
    """Ordinary main-answer majority vote, ties broken by first proposal order."""
    votes = [s for s in streams if s.answer is not None]
    if not votes:
        return None
    candidates = candidate_set(task, None, streams)
    counts = {c: sum(task.equivalent(s.answer, c) for s in votes) for c in candidates}
    return max(candidates, key=lambda c: (counts[c], -candidates.index(c)))


def _select(candidates: Sequence[str], distribution: Dict[str, float]) -> Prediction:
    winner = max(candidates, key=lambda c: (distribution.get(c, 0.0), -candidates.index(c)))
    return Prediction(distribution.get(winner, 0.0), answer=winner,
                      details={"other": distribution.get(OTHER, 0.0)})
