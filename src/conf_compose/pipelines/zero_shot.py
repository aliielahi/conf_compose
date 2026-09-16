"""Zero-shot pipeline: answer, grade, and attach every configured confidence signal to each example."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from conf_compose.confidence_estimators import (ConsistencyConfidence, SelfVerification, SequenceProbability,
                                   VerbalizedConfidence)
from conf_compose.constants import SAMPLING, SEQUENCE_PROBABILITY


@dataclass
class ZeroShotConfig:
    max_tokens: int = 1024
    scopes: Tuple[str, ...] = tuple(SEQUENCE_PROBABILITY["scopes"])
    tail_fraction: float = SEQUENCE_PROBABILITY["tail_fraction"]
    debias: bool = False
    verbalized: bool = True
    verbal_temperature: float = SAMPLING["verbal_temperature"]
    verbal_repeats: int = SAMPLING["verbal_repeats"]
    verification: bool = True
    consistency_temperatures: Tuple[float, ...] = (SAMPLING["consistency_temperature"],)
    consistency_samples: int = SAMPLING["consistency_samples"]
    top_p: float = SAMPLING["top_p"]
    top_k: int = SAMPLING["top_k"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_zero_shot(llm, task, examples, config: Optional[ZeroShotConfig] = None,
                  timings: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    config = config or ZeroShotConfig()
    timings = {} if timings is None else timings
    prompts = [task.prompt(example) for example in examples]
    with _timed(timings, "generation"):
        generations = llm.generate(prompts, max_tokens=config.max_tokens, temperature=0.0)
    responses = [generation.text for generation in generations]
    records = [_record(task, example, generation) for example, generation in zip(examples, generations, strict=True)]

    tail = f"tail{round(config.tail_fraction * 100)}"
    for scope in config.scopes:
        estimator = SequenceProbability(llm, scope=scope, debias=config.debias, tail_fraction=config.tail_fraction)
        signals = {f"seq_{scope}": "confidence", f"seq_{scope}_min": "min_confidence",
                   f"seq_{scope}_{tail}": "tail_confidence"}
        if config.debias:
            signals[f"seq_{scope}_debiased"] = "debiased_confidence"
        with _timed(timings, f"seq_{scope}"):
            results = estimator.estimate(task, examples, responses)
        for record, result in zip(records, results):
            record["confidence"].update({name: getattr(result, attr) if result else None
                                         for name, attr in signals.items()})
            record.setdefault("token_logprobs", {})[scope] = result.token_logprobs if result else None

    if config.verification:
        with _timed(timings, "verification"):
            values = SelfVerification(llm).estimate(task, examples, responses)
        for record, value in zip(records, values):
            record["confidence"]["verification"] = value

    if config.verbalized:
        estimator = VerbalizedConfidence(llm, repeats=config.verbal_repeats, temperature=config.verbal_temperature,
                                         top_p=config.top_p, top_k=config.top_k)
        with _timed(timings, "verbalized"):
            results = estimator.estimate(prompts, responses)
        for record, result in zip(records, results):
            record["confidence"]["verbalized"] = result.confidence
            record["verbalized_raw"] = result.raw

    predictions = [record["prediction"] for record in records]
    for temperature in config.consistency_temperatures:
        estimator = ConsistencyConfidence(llm, config.consistency_samples, temperature, config.top_p, config.top_k,
                                          config.max_tokens)
        name = f"consistency_t{temperature:g}"
        with _timed(timings, name):
            results = estimator.estimate(task, examples, predictions)
        for record, result in zip(records, results):
            record["confidence"].update({name: result.agreement, f"{name}_margin": result.margin,
                                         f"{name}_entropy": result.entropy_confidence})
            record.setdefault("sampled_answers", {})[name] = result.answers
    return records


@contextmanager
def _timed(timings: Dict[str, float], stage: str):
    start = time.time()
    yield
    timings[stage] = time.time() - start


def signal_names(records: Sequence[Dict[str, Any]]) -> List[str]:
    return list(records[0]["confidence"]) if records else []


def signal(records: Sequence[Dict[str, Any]], name: str, missing: float = 0.0) -> List[float]:
    return [missing if record["confidence"][name] is None else record["confidence"][name] for record in records]


def _record(task, example, generation) -> Dict[str, Any]:
    answer = task.extract_answer(generation.text)
    prediction = answer.text if answer else None
    return {
        "id": example.id,
        "gold": example.answer,
        "prediction": prediction,
        "correct": task.is_correct(prediction, example),
        "explicit_answer": bool(answer and answer.explicit),
        "finish_reason": generation.finish_reason,
        "response": generation.text,
        "confidence": {},
    }
