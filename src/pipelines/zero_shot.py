"""Zero-shot pipeline: answer, grade, and attach every confidence signal to each example."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from confidence_estimators import SequenceProbability, VerbalizedConfidence


def run_zero_shot(llm, task, examples, max_tokens: int = 512, scopes: Sequence[str] = ("answer_no_reasoning",),
                  debias: bool = True, verbalized: bool = True) -> List[Dict[str, Any]]:
    prompts = [task.prompt(example) for example in examples]
    responses = llm(prompts, max_tokens=max_tokens, temperature=0.0)
    answers = [task.extract_answer(response) for response in responses]

    records = []
    for example, response, answer in zip(examples, responses, answers):
        prediction = answer.text if answer else None
        records.append({
            "id": example.id,
            "gold": example.answer,
            "prediction": prediction,
            "correct": task.is_correct(prediction, example),
            "response": response,
            "confidence": {},
        })

    for scope in scopes:
        results = SequenceProbability(llm, scope=scope, debias=debias).estimate(task, examples, responses)
        for record, result in zip(records, results):
            record["confidence"][f"seq_{scope}"] = result.confidence if result else None
            if debias:
                record["confidence"][f"seq_{scope}_debiased"] = result.debiased_confidence if result else None

    if verbalized:
        for record, result in zip(records, VerbalizedConfidence(llm).estimate(prompts, responses)):
            record["confidence"]["verbalized"] = result.confidence
            record["verbalized_raw"] = result.raw
    return records


def signal_names(records: Sequence[Dict[str, Any]]) -> List[str]:
    return list(records[0]["confidence"]) if records else []


def signal(records: Sequence[Dict[str, Any]], name: str, missing: float = 0.0) -> List[float]:
    values: List[Optional[float]] = [record["confidence"][name] for record in records]
    return [missing if value is None else value for value in values]
