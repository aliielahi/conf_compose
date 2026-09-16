"""Zero-shot pipeline: answer, grade, and attach every confidence signal to each example."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from confidence_estimators import ConsistencyConfidence, SequenceProbability, VerbalizedConfidence


def run_zero_shot(llm, task, examples, max_tokens: int = 512, scopes: Sequence[str] = ("response",),
                  debias: bool = True, tail_fraction: float = 0.1, verbalized: bool = True,
                  verbal_temperature: float = 0.3, consistency_temperatures: Sequence[float] = (),
                  consistency_samples: int = 5) -> List[Dict[str, Any]]:
    prompts = [task.prompt(example) for example in examples]
    generations = llm.generate(prompts, max_tokens=max_tokens, temperature=0.0)
    responses = [generation.text for generation in generations]

    records = []
    for example, generation in zip(examples, generations, strict=True):
        answer = task.extract_answer(generation.text)
        prediction = answer.text if answer else None
        records.append({
            "id": example.id,
            "gold": example.answer,
            "prediction": prediction,
            "correct": task.is_correct(prediction, example),
            "explicit_answer": bool(answer and answer.explicit),
            "finish_reason": generation.finish_reason,
            "response": generation.text,
            "confidence": {},
        })

    seq_signals = {"": "confidence", "_min": "min_confidence", f"_tail{round(tail_fraction * 100)}": "tail_confidence"}
    if debias:
        seq_signals["_debiased"] = "debiased_confidence"
    for scope in scopes:
        estimator = SequenceProbability(llm, scope=scope, debias=debias, tail_fraction=tail_fraction)
        for record, result in zip(records, estimator.estimate(task, examples, responses)):
            for suffix, attribute in seq_signals.items():
                record["confidence"][f"seq_{scope}{suffix}"] = getattr(result, attribute) if result else None
            record.setdefault("token_logprobs", {})[scope] = result.token_logprobs if result else None

    if verbalized:
        estimator = VerbalizedConfidence(llm, temperature=verbal_temperature)
        for record, result in zip(records, estimator.estimate(prompts, responses)):
            record["confidence"]["verbalized"] = result.confidence
            record["verbalized_raw"] = result.raw

    predictions = [record["prediction"] for record in records]
    for temperature in consistency_temperatures:
        estimator = ConsistencyConfidence(llm, consistency_samples, temperature, max_tokens)
        for record, result in zip(records, estimator.estimate(task, examples, predictions)):
            record["confidence"][f"consistency_t{temperature:g}"] = result.confidence
            record.setdefault("sampled_answers", {})[f"t{temperature:g}"] = result.answers
    return records


def signal_names(records: Sequence[Dict[str, Any]]) -> List[str]:
    return list(records[0]["confidence"]) if records else []


def signal(records: Sequence[Dict[str, Any]], name: str, missing: float = 0.0) -> List[float]:
    values: List[Optional[float]] = [record["confidence"][name] for record in records]
    return [missing if value is None else value for value in values]
