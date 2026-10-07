"""Synchronous debate round for one loaded model: every group it sits in, every question, one batch per task.

Round r reads every agent's round r-1 records (round 0 from the inference store), generates this model's
revisions, then runs round 0's own confidence pipeline on the exact context the agent saw.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

from conf_compose.confidence_estimators import Target
from conf_compose.data import Example, get_task
from conf_compose.pipelines.confidence import ConfidenceConfig, estimate_confidence
from conf_compose.pipelines.inference import STORE

from .prompts import PEER_LABELS
from .revision import Peer, agreement, peer_order, resolve_answer, revision_messages
from .settings import DEBATE_STORE, DebateSettings
from .store import load_round, round0_confidence_config, shared_ids, write_rows

# Below this many output tokens a turn cannot answer, so it is recorded as an overflow instead.
MIN_OUTPUT = 256


def confidence_config(settings: DebateSettings, model: str, store: Path = STORE) -> ConfidenceConfig:
    """Round 0's estimator settings for this model, minus the verbalized and verification signals."""
    saved = round0_confidence_config(settings.task, model, settings.voter, settings.match, store)
    fields = {key: tuple(value) if isinstance(value, list) else value for key, value in saved.items()
              if key in ConfidenceConfig.__dataclass_fields__}
    return ConfidenceConfig(**{**fields, "verbalized": False, "verification": False,
                               "verification_context": False, "max_tokens": settings.max_tokens})


def build_turns(task, settings: DebateSettings, model: str, round_index: int, store: Path,
                out: Path) -> List[Dict[str, Any]]:
    """Every question this agent revises in this group, with the messages and the peers it is shown."""
    previous = {member: load_round(settings, member, round_index - 1, store, out) for member in settings.group}
    turns = []
    for example_id in shared_ids(previous, settings.limit):
        own = previous[model][example_id]
        peers = [Peer(member, previous[member][example_id]["prediction"], previous[member][example_id]["response"])
                 for member in settings.group if member != model]
        shown = peer_order(peers, settings.seed, settings.task, example_id, model, round_index)
        example = Example(example_id, own["question"], own["gold"],
                          {"choices": own.get("choices") or {}, "options": own.get("options")})
        if task.prompt(example) != own["prompt"]:
            raise ValueError(f"{settings.task}/{example_id}: the task prompt no longer matches round 0's")
        turns.append({"settings": settings, "model": model, "example": example, "own": own, "peers": shown,
                      "messages": revision_messages(task, own["prompt"], own["response"], shown)})
    return turns


def budgets(llm, turns: Sequence[Dict[str, Any]], max_tokens: int, context: int) -> None:
    """Count each prompt with the agent's own tokenizer; shrink the output budget to fit, never the input."""
    rendered = [llm.render(None, turn["messages"]) for turn in turns]
    for turn, ids in zip(turns, llm.encode(rendered)["input_ids"] if rendered else []):
        turn["prompt_tokens"] = len(ids)
        room = context - len(ids)
        turn["budget"] = max_tokens if room >= max_tokens else (room // 64) * 64
        turn["overflow"] = turn["budget"] < MIN_OUTPUT


def generate_round(llm, model: str, jobs: Sequence[DebateSettings], round_index: int, context: int,
                   store: Path = STORE, out: Path = DEBATE_STORE) -> Dict[str, int]:
    """Write this model's round for every job; returns turns written per cell name."""
    written = {}
    by_scope = defaultdict(list)
    for settings in jobs:
        by_scope[(settings.task, settings.decoding)].append(settings)
    for (task_name, decoding), cells in by_scope.items():
        task = get_task(task_name)
        turns = [turn for settings in cells for turn in build_turns(task, settings, model, round_index, store, out)]
        budgets(llm, turns, cells[0].max_tokens, context)
        config = confidence_config(cells[0], model, store)
        scope = f"debate:{task_name}:{decoding}:r{round_index}"
        for budget in sorted({turn["budget"] for turn in turns if not turn["overflow"]}):
            batch = [turn for turn in turns if not turn["overflow"] and turn["budget"] == budget]
            print(f"  {task_name} round {round_index}: {len(batch)} turn(s) at {budget} tokens", flush=True)
            _revise(llm, task, batch, budget, cells[0].temperature, scope, replace(config, max_tokens=budget))
        overflow = sum(turn["overflow"] for turn in turns)
        if overflow:
            print(f"  {task_name} round {round_index}: {overflow} turn(s) exceed the {context}-token context",
                  flush=True)
        for settings in cells:
            rows = [_record(task, turn, round_index) for turn in turns if turn["settings"] is settings]
            write_rows(settings.round_path(model, round_index, out), rows)
            written[settings.name] = len(rows)
    return written


def _revise(llm, task, batch, budget: int, temperature: float, scope: str, config: ConfidenceConfig) -> None:
    """The revision is sampled in its own execution scope, so it is never also one of its consistency draws."""
    llm.execution = f"{scope}:answer"
    generations = llm.generate([turn["messages"] for turn in batch], max_tokens=budget, temperature=temperature,
                               logprobs=True)
    targets = []
    for turn, generation in zip(batch, generations):
        answer, source = resolve_answer(task, generation.text, turn["own"]["prediction"], turn["peers"])
        turn.update(generation=generation, answer=answer, source=source)
        targets.append(Target(example=turn["example"], conversation=turn["messages"], response=generation.text,
                              answer=answer, logprobs=generation.logprobs))
    llm.execution = scope
    for turn, output in zip(batch, estimate_confidence(llm, task, targets, config)):
        turn["confidence"], turn["details"] = output.signals, output.details


def _record(task, turn: Dict[str, Any], round_index: int) -> Dict[str, Any]:
    """The inference-store record schema, plus what the agent saw and how its answer was resolved."""
    own, example, peers = turn["own"], turn["example"], turn["peers"]
    generation = turn.get("generation")
    answer = turn.get("answer")
    record = {
        "id": example.id, "question": own["question"], "prompt": own["prompt"], "options": own.get("options"),
        "choices": own.get("choices"), "gold": own["gold"], "prediction": answer,
        "correct": task.is_correct(answer, example), "explicit_answer": turn.get("source") == "explicit",
        "finish_reason": generation.finish_reason if generation else None,
        "response": generation.text if generation else "", "confidence": turn.get("confidence", {}),
        **turn.get("details", {}), "token_budget": turn["budget"],
        "round": round_index, "agent": turn["settings"].group.index(turn["model"]), "model": turn["model"],
        "group": list(turn["settings"].group), "messages": turn["messages"],
        "peer_order": [peer.model for peer in peers], "peer_labels": list(PEER_LABELS[:len(peers)]),
        "answer_source": turn.get("source", "none"), **agreement(task, answer, own["prediction"], peers),
        "prompt_tokens": turn["prompt_tokens"], "output_tokens": generation.output_tokens if generation else None,
        "error": "context_overflow" if turn["overflow"] else (generation.error if generation else None),
    }
    return record
