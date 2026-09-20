"""Zero-shot pipeline: answer each example once, grade it, and attach every configured confidence signal."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence

from conf_compose.confidence_estimators import zero_shot_targets

from .confidence import ConfidenceConfig, estimate_confidence, _timed

ZeroShotConfig = ConfidenceConfig


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
    records = [_record(task, example, generation, target)
               for example, generation, target in zip(examples, generations, targets, strict=True)]
    for record, output in zip(records, estimate_confidence(llm, task, targets, config, timings)):
        record["confidence"] = output.signals
        record.update(output.details)
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


def _record(task, example, generation, target) -> Dict[str, Any]:
    answer = task.extract_answer(generation.text)
    return {
        "id": example.id,
        "gold": example.answer,
        "prediction": target.answer,
        "correct": task.is_correct(target.answer, example),
        "explicit_answer": bool(answer and answer.explicit),
        "finish_reason": generation.finish_reason,
        "response": generation.text,
        "confidence": {},
    }
