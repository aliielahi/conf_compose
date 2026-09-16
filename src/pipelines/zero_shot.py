"""Zero-shot pipeline: answer, grade, and attach every configured confidence signal to each example."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from confidence_estimators import (ConsistencyConfidence, SelfVerification, SequenceProbability,
                                   VerbalizedConfidence)


@dataclass
class ZeroShotConfig:
    max_tokens: int = 512
    scopes: Tuple[str, ...] = ("response",)
    tail_fraction: float = 0.1
    debias: bool = False
    verbalized: bool = True
    verbal_temperature: float = 1.0
    verbal_repeats: int = 3
    verification: bool = True
    consistency_temperatures: Tuple[float, ...] = (0.7,)
    consistency_samples: int = 10
    top_p: float = 1.0
    top_k: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_zero_shot(llm, task, examples, config: Optional[ZeroShotConfig] = None) -> List[Dict[str, Any]]:
    config = config or ZeroShotConfig()
    prompts = [task.prompt(example) for example in examples]
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
        for record, result in zip(records, estimator.estimate(task, examples, responses)):
            record["confidence"].update({name: getattr(result, attr) if result else None
                                         for name, attr in signals.items()})
            record.setdefault("token_logprobs", {})[scope] = result.token_logprobs if result else None

    if config.verification:
        for record, value in zip(records, SelfVerification(llm).estimate(task, examples, responses)):
            record["confidence"]["verification"] = value

    if config.verbalized:
        estimator = VerbalizedConfidence(llm, repeats=config.verbal_repeats, temperature=config.verbal_temperature,
                                         top_p=config.top_p, top_k=config.top_k)
        for record, result in zip(records, estimator.estimate(prompts, responses)):
            record["confidence"]["verbalized"] = result.confidence
            record["verbalized_raw"] = result.raw

    predictions = [record["prediction"] for record in records]
    for temperature in config.consistency_temperatures:
        estimator = ConsistencyConfidence(llm, config.consistency_samples, temperature, config.top_p, config.top_k,
                                          config.max_tokens)
        name = f"consistency_t{temperature:g}"
        for record, result in zip(records, estimator.estimate(task, examples, predictions)):
            record["confidence"].update({name: result.agreement, f"{name}_margin": result.margin,
                                         f"{name}_entropy": result.entropy_confidence})
            record.setdefault("sampled_answers", {})[name] = result.answers
    return records


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
