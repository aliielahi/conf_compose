"""Zero-shot pipeline: answer each example once, grade it, and attach every configured confidence signal."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

from conf_compose.confidence_estimators import zero_shot_targets

from .confidence import ConfidenceConfig, estimate_confidence, _timed

ZeroShotConfig = ConfidenceConfig

# Bump when a record gains or loses a field, so a cell's contents are identifiable after the fact.
RECORD_SCHEMA = 5


def run_zero_shot(llm, task, examples, config: Optional[ConfidenceConfig] = None,
                  timings: Optional[Dict[str, Dict[str, float]]] = None) -> List[Dict[str, Any]]:
    config = config or ConfidenceConfig()
    timings = {} if timings is None else timings
    prompts = [task.prompt(example) for example in examples]
    with _timed(timings, "generation", getattr(llm, "cache", None)):
        with _answer_scope(llm, config.answer_temperature):
            generations = llm.generate(prompts, max_tokens=config.max_tokens,
                                       temperature=config.answer_temperature, logprobs=True)

    targets = zero_shot_targets(task, examples, generations)
    records = [_record(task, example, generation, target, prompt)
               for example, generation, target, prompt in zip(examples, generations, targets, prompts,
                                                              strict=True)]
    for record, output in zip(records, estimate_confidence(llm, task, targets, config, timings)):
        record["confidence"] = output.signals
        record.update(output.details)
        record["token_budget"] = config.max_tokens
    return records


def retry_truncated(llm, task, examples, records, config: ConfidenceConfig, max_tokens: int,
                    timings: Optional[Dict[str, Dict[str, float]]] = None) -> List[Dict[str, Any]]:
    """Regenerate only the rows that hit the ceiling, at a larger budget, signals and resamples included."""
    indices = [i for i, record in enumerate(records) if record["finish_reason"] == "length"]
    print(f"    retry: {len(indices)}/{len(records)} row(s) truncated, regenerating at {max_tokens} tokens",
          flush=True)
    if not indices:
        return records
    redone = run_zero_shot(llm, task, [examples[i] for i in indices],
                           replace(config, max_tokens=max_tokens), timings)
    for index, record in zip(indices, redone):
        records[index] = record
    still = sum(record["finish_reason"] == "length" for record in records)
    print(f"    retry: {still}/{len(records)} still truncated ({100 * still / len(records):.1f}%)", flush=True)
    return records


@contextmanager
def _answer_scope(llm, temperature: float):
    """A sampled answer gets its own execution scope, so it is never also a consistency draw."""
    original = getattr(llm, "execution", "")
    if temperature > 0:
        llm.execution = f"{original}:answer"
    try:
        yield
    finally:
        llm.execution = original


def signal_names(records: Sequence[Dict[str, Any]]) -> List[str]:
    return list(records[0]["confidence"]) if records else []


def signal(records: Sequence[Dict[str, Any]], name: str, missing: float = 0.0) -> List[float]:
    return [missing if record["confidence"][name] is None else record["confidence"][name] for record in records]


def _record(task, example, generation, target, prompt: str) -> Dict[str, Any]:
    answer = task.extract_answer(generation.text)
    return {
        "id": example.id,
        "question": example.question,
        "prompt": prompt,
        "options": example.meta.get("options") or sorted(example.meta.get("choices", {})) or None,
        "choices": example.meta.get("choices") or None,
        "gold": example.answer,
        "prediction": target.answer,
        "correct": task.is_correct(target.answer, example),
        "explicit_answer": bool(answer and answer.explicit),
        "finish_reason": generation.finish_reason,
        "response": generation.text,
        "confidence": {},
    }
