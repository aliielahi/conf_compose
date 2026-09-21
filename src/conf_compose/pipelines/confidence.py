"""Run every configured estimator over targets, whether they come from zero-shot answers or debate turns."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from conf_compose.confidence_estimators import (ConsistencyConfidence, SelfVerification, SequenceProbability,
                                                Target, VerbalizedConfidence, results_from_logprobs)
from conf_compose.constants import SAMPLING, SEQUENCE_PROBABILITY


@dataclass
class ConfidenceConfig:
    max_tokens: int = 1024
    answer_temperature: float = 0.0
    scopes: Tuple[str, ...] = tuple(SEQUENCE_PROBABILITY["scopes"])
    tail_fraction: float = SEQUENCE_PROBABILITY["tail_fraction"]
    debias: bool = False
    verbalized: bool = True
    verbal_temperature: float = SAMPLING["verbal_temperature"]
    verbal_repeats: int = SAMPLING["verbal_repeats"]
    verification: bool = True
    verification_context: bool = True
    consistency_temperatures: Tuple[float, ...] = (SAMPLING["consistency_temperature"],)
    consistency_samples: int = SAMPLING["consistency_samples"]
    top_p: float = SAMPLING["top_p"]
    top_k: int = SAMPLING["top_k"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConfidenceOutput:
    signals: Dict[str, Optional[float]]
    details: Dict[str, Any]


def estimate_confidence(llm, task, targets: Sequence[Target], config: Optional[ConfidenceConfig] = None,
                        timings: Optional[Dict[str, Dict[str, float]]] = None) -> List[ConfidenceOutput]:
    config = config or ConfidenceConfig()
    timings = {} if timings is None else timings
    cache = getattr(llm, "cache", None)
    outputs = [ConfidenceOutput({}, {}) for _ in targets]
    if not targets:
        return outputs

    tail = f"tail{round(config.tail_fraction * 100)}"
    for scope in config.scopes:
        names = {f"seq_{scope}": "confidence", f"seq_{scope}_min": "min_confidence",
                 f"seq_{scope}_{tail}": "tail_confidence"}
        if config.debias:
            names[f"seq_{scope}_debiased"] = "debiased_confidence"
        with _timed(timings, f"seq_{scope}", cache):
            reusable = scope == "response" and not config.debias and all(t.logprobs for t in targets if t.response)
            if reusable:
                results = results_from_logprobs([t.logprobs for t in targets], config.tail_fraction)
            else:
                results = SequenceProbability(llm, scope, config.debias, config.tail_fraction).estimate(task, targets)
        for output, result in zip(outputs, results):
            output.signals.update({name: getattr(result, attr) if result else None for name, attr in names.items()})
            output.details.setdefault("token_logprobs", {})[scope] = result.token_logprobs if result else None

    for name, in_context in (("verification", False), ("verification_context", True)):
        if not (config.verification if not in_context else config.verification_context):
            continue
        estimator = SelfVerification(llm, context=in_context)
        with _timed(timings, name, cache):
            values = estimator.estimate(task, targets)
        _check_stage(name, values, targets,
                     f"top tokens were {getattr(estimator, 'sample_tokens', [])}; a reasoning model may need "
                     f"CHAT_TEMPLATE_KWARGS to turn thinking off")
        for output, value in zip(outputs, values):
            output.signals[name] = value

    if config.verbalized:
        estimator = VerbalizedConfidence(llm, repeats=config.verbal_repeats, temperature=config.verbal_temperature,
                                         top_p=config.top_p, top_k=config.top_k)
        with _timed(timings, "verbalized", cache):
            results = estimator.estimate(task, targets)
        _check_stage("verbalized", [result.confidence for result in results], targets)
        for output, result in zip(outputs, results):
            output.signals["verbalized"] = result.confidence
            output.details["verbalized_raw"] = result.raw

    for temperature in config.consistency_temperatures:
        estimator = ConsistencyConfidence(llm, config.consistency_samples, temperature, config.top_p, config.top_k,
                                          config.max_tokens)
        name = f"consistency_t{temperature:g}"
        with _timed(timings, name, cache):
            results = estimator.estimate(task, targets)
        _check_stage(name, [result.valid_fraction or None for result in results], targets)
        for output, result in zip(outputs, results):
            output.signals.update({name: result.agreement, f"{name}_margin": result.margin,
                                   f"{name}_entropy": result.entropy_confidence})
            output.details.setdefault("sampled_answers", {})[name] = result.answers
            output.details.setdefault("sampled_responses", {})[name] = result.responses
    return outputs


def _check_stage(stage: str, values: Sequence[Optional[float]], targets: Sequence[Target],
                 hint: str = "") -> None:
    """An estimator that returns nothing for every answerable target means the backend failed."""
    answerable = [i for i, target in enumerate(targets) if target.answer is not None]
    if answerable and all(values[i] is None for i in answerable):
        raise RuntimeError(f"{stage}: no value for any of {len(answerable)} answerable targets"
                           + (f"; {hint}" if hint else "; check the backend"))


@contextmanager
def _timed(timings: Dict[str, Dict[str, float]], stage: str, cache=None):
    start = time.time()
    hits, misses = (cache.hits, cache.misses) if cache else (0, 0)
    yield
    timings[stage] = {"seconds": time.time() - start,
                      "cache_hits": (cache.hits - hits) if cache else 0,
                      "cache_misses": (cache.misses - misses) if cache else 0}
