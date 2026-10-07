"""Candidate scoring after the last round: every agent scores every answer anyone proposed, in each round's context."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from conf_compose.composition.candidates import label_space
from conf_compose.confidence_estimators.sequence_prob import Candidate, CandidateScorer
from conf_compose.data import get_task
from conf_compose.pipelines.inference import STORE

from .revision import answer_span
from .settings import DEBATE_STORE, DebateSettings
from .store import load_round, shared_ids, write_rows

SAMPLES = "consistency_t0.7"


class ContextTask:
    """A task whose prompt is a recorded conversation, so the shared scorer conditions on what the agent saw."""

    def __init__(self, task):
        self.task = task

    @property
    def answer_prefix(self) -> str:
        return self.task.answer_prefix

    def prompt(self, messages):
        return messages

    def null_prompt(self, null_input: str) -> str:
        return self.task.null_prompt(null_input)


def union_pool(task, rounds: Sequence[Dict[str, Dict[str, Any]]], example_id: str) -> List[Candidate]:
    """Labels first, then every agent's answer in every round, then their resamples: the store's pool rule."""
    first = next(iter(rounds[0].values()))[example_id]
    choices = first.get("choices") or {label: None for label in (label_space(task) or ())}
    pool = [Candidate(label, label, text, "labels") for label, text in sorted(choices.items())]
    generated = [records[example_id]["prediction"] for agents in rounds for records in agents.values()]
    resampled = [answer for agents in rounds for records in agents.values()
                 for answer in (records[example_id].get("sampled_answers") or {}).get(SAMPLES, [])]
    for source, answers in (("generated", generated), ("resampled", resampled)):
        for answer in answers:
            if answer is None:
                continue
            existing = next((candidate for candidate in pool if task.equivalent(candidate.answer, answer)), None)
            if existing is None:
                pool.append(Candidate(answer, answer, None, source))
            elif source not in existing.source:
                existing.source = f"{existing.source}+{source}"
    return pool


def context(record: Dict[str, Any]):
    """Round 0 saw only the question; a revision saw its full recorded conversation."""
    return record.get("messages") or [{"role": "user", "content": record["prompt"]}]


def score_candidates(llm, model: str, jobs: Sequence[DebateSettings], through: int, store: Path = STORE,
                     out: Path = DEBATE_STORE) -> Dict[str, int]:
    """Write this model's candidate scores for rounds 0..through of every job; returns rows per cell."""
    written = {}
    by_task = defaultdict(list)
    for settings in jobs:
        by_task[settings.task].append(settings)
    for task_name, cells in by_task.items():
        task = get_task(task_name)
        scorer = CandidateScorer(llm, debias=True)
        for settings in cells:
            rounds = [{member: load_round(settings, member, r, store, out) for member in settings.group}
                      for r in range(through + 1)]
            ids = shared_ids(rounds[through])
            requests = [(r, example_id, rounds[r][model][example_id]) for example_id in ids
                        for r in range(through + 1) if not rounds[r][model][example_id].get("error")]
            pools = {example_id: union_pool(task, rounds, example_id) for example_id in ids}
            spans = [answer_span(task, record["response"]) for _, _, record in requests]
            entries = scorer.score(ContextTask(task), [context(record) for _, _, record in requests],
                                   [record["response"] for _, _, record in requests],
                                   [span.start if span else None for span in spans],
                                   [pools[example_id] for _, example_id, _ in requests])
            rows = [{"id": example_id, "round": r, "candidate_scores": entry}
                    for (r, example_id, _), entry in zip(requests, entries)]
            print(f"  {task_name} {settings.name}: {len(rows)} scored turn(s), "
                  f"{sum(len(p) for p in pools.values()) / max(len(pools), 1):.1f} candidates per question",
                  flush=True)
            write_rows(settings.scores_path(model, through, out), rows)
            written[settings.name] = len(rows)
    return written
