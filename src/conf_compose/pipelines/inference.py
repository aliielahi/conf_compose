"""The shared inference store: one model's answers and confidence signals, generated once and reused."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from conf_compose.composition.candidates import label_space
from conf_compose.confidence_estimators.sequence_prob import Candidate, CandidateScorer
from conf_compose.constants import RESULTS_DIR, SAMPLING, TASKS

from conf_compose.utils.llm_calls import LLM

from .runs import split_summary
from .zero_shot import RECORD_SCHEMA, ZeroShotConfig, retry_truncated, run_zero_shot

STORE = RESULTS_DIR / "inferences"

# Options added after cells were written; at their default they stay out of the digest so old cells resolve.
ADDITIVE_DEFAULTS = {"retry_max_tokens": None, "candidate_scores": False, "candidate_models": None,
                     "candidate_samples": False, "consistency_logprobs": False}


def load_model(model: str, cache_dir=None, **overrides):
    """One loaded model, with the local-engine defaults and any template mode this alias needs."""
    from conf_compose.constants import CACHE_DIR, HF, VLLM
    from conf_compose.utils.llm_calls.hf_models import CHAT_TEMPLATE_KWARGS, GENERATION_PREFIX

    cache_dir = str(cache_dir or CACHE_DIR)
    alias = model.split("/")[-1]
    if CHAT_TEMPLATE_KWARGS.get(alias):
        overrides.setdefault("chat_template_kwargs", CHAT_TEMPLATE_KWARGS[alias])
    if GENERATION_PREFIX.get(alias):
        overrides.setdefault("generation_prefix", GENERATION_PREFIX[alias])
    if model.startswith("hf/"):
        return LLM(model, cache_dir=cache_dir, batch_size=HF["batch_size"], **overrides)
    engine = {"max_num_seqs": VLLM["max_num_seqs"], "max_num_batched_tokens": VLLM["max_num_batched_tokens"]}
    settings = {"gpu_memory_utilization": VLLM["gpu_memory_utilization"],
                "max_model_len": VLLM["max_model_len"], "quiet": True, **overrides}
    return LLM(model, cache_dir=cache_dir, engine_kwargs=engine, **settings)


@dataclass(frozen=True)
class InferenceSettings:
    """Everything that changes what a model writes; equal digests mean two inferences are interchangeable."""
    task: str
    model: str
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    max_tokens: Optional[int] = None
    answer_temperature: float = 0.0
    voter: int = 0
    consistency_samples: int = SAMPLING["consistency_samples"]
    consistency_temperature: float = SAMPLING["consistency_temperature"]
    verbalized: bool = True
    verification: bool = True
    verification_context: bool = False
    debias: bool = False
    retry_max_tokens: Optional[int] = None
    candidate_scores: bool = False
    candidate_models: Optional[Tuple[str, ...]] = None
    candidate_samples: bool = False
    consistency_logprobs: bool = False

    def filled(self) -> "InferenceSettings":
        """Task defaults applied; None means the default size and 0 means the whole split, unsampled."""
        defaults = TASKS[self.task]
        return replace(self, n_val=defaults["n_val"] if self.n_val is None else self.n_val,
                       n_test=defaults["n_test"] if self.n_test is None else self.n_test,
                       max_tokens=self.max_tokens or defaults["max_tokens"])

    @property
    def decoding(self) -> str:
        if self.answer_temperature <= 0:
            return "greedy"
        return f"s{int(round(self.answer_temperature * 100)):02d}v{self.voter}"

    @property
    def size(self) -> str:
        n_test, n_val = self.filled().n_test, self.filled().n_val
        return ("full" if n_test == -1 else f"n{n_test}") + ("" if n_val else "_noval")

    @property
    def tag(self) -> str:
        retry = f"_r{self.retry_max_tokens}" if self.retry_max_tokens else ""
        scored = f"_cs{len(self.candidate_models or ())}" if self.candidate_scores else ""
        scored += "s" if self.candidate_scores and self.candidate_samples else ""
        return (f"{self.model.replace('/', '__')}--{self.decoding}"
                f"_k{self.consistency_samples}_{self.size}{retry}{scored}"
                + ("_lp" if self.consistency_logprobs else ""))

    def base(self) -> "InferenceSettings":
        """The same cell without candidate scoring, which is where the candidate answers come from."""
        return replace(self.filled(), candidate_scores=False, candidate_models=None,
                       candidate_samples=False)

    @property
    def digest(self) -> str:
        """Options left at an additive default are omitted, so adding one never moves an existing cell."""
        payload = {key: value for key, value in asdict(self.filled()).items()
                   if value is not None and ADDITIVE_DEFAULTS.get(key, object()) != value}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self.filled()), "digest": self.digest}


def inference_dir(settings: InferenceSettings, store: Path = STORE) -> Path:
    settings = settings.filled()
    return Path(store) / settings.task / f"{settings.tag}--{settings.digest}"


def exists(settings: InferenceSettings, store: Path = STORE) -> bool:
    return (inference_dir(settings, store) / "settings.json").exists()


def missing(wanted: Sequence[InferenceSettings], store: Path = STORE) -> List[InferenceSettings]:
    return [settings for settings in wanted if not exists(settings, store)]


def by_model(wanted: Sequence[InferenceSettings]) -> Dict[str, List[InferenceSettings]]:
    """Group requests so a runner loads each model once, whatever tasks and voters it owes."""
    grouped: Dict[str, List[InferenceSettings]] = {}
    for settings in wanted:
        grouped.setdefault(settings.model, []).append(settings)
    return grouped


def ensure_inference(settings: InferenceSettings, llm, task, store: Path = STORE) -> Path:
    """Generate this inference unless the store already holds it; the llm is loaded by the caller."""
    settings = settings.filled()
    out_dir = inference_dir(settings, store)
    if exists(settings, store):
        return out_dir

    config = ZeroShotConfig(max_tokens=settings.max_tokens, answer_temperature=settings.answer_temperature,
                            verbalized=settings.verbalized, verification=settings.verification,
                            verification_context=settings.verification_context, debias=settings.debias,
                            consistency_samples=settings.consistency_samples,
                            consistency_temperatures=(settings.consistency_temperature,),
                            consistency_logprobs=settings.consistency_logprobs)
    llm.execution = f"{settings.task}:{settings.decoding}"
    validation = [] if settings.n_val == 0 else task.load("validation", n=_count(settings.n_val))
    test = [] if settings.n_test == 0 else task.load("test", n=_count(settings.n_test))

    start, timings = time.time(), {}
    examples = validation + test
    combined = run_zero_shot(llm, task, examples, config, timings)
    if settings.retry_max_tokens:
        combined = retry_truncated(llm, task, examples, combined, config, settings.retry_max_tokens, timings)
    if settings.candidate_scores:
        add_candidate_scores(llm, task, examples, combined, settings, store)
    records = {"validation": combined[:len(validation)], "test": combined[len(validation):]}

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        if rows:
            (out_dir / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (out_dir / "settings.json").write_text(json.dumps(
        {"settings": settings.to_dict(), "record_schema": RECORD_SCHEMA, "config": config.to_dict(),
         "execution": llm.execution,
         "splits": {split: split_summary(rows) for split, rows in records.items() if rows},
         "timings": timings, "seconds": round(time.time() - start, 1)}, indent=2))
    return out_dir


def candidate_pool(task, settings: InferenceSettings, store: Path = STORE) -> Dict[str, List[Candidate]]:
    """Every distinct answer any model produced, its resamples if asked, plus the full label set if one exists."""
    generated: Dict[str, List[str]] = {}
    sampled: Dict[str, List[str]] = {}
    options: Dict[str, Dict[str, Any]] = {}
    seen, wanted = [], []
    for model in settings.candidate_models or (settings.model,):
        source = replace(settings.base(), model=model)
        for split in ("validation", "test"):
            path = records_path(source, split, store)
            wanted.append(path)
            if not path.exists():
                continue
            seen.append(path)
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("choices"):
                    options.setdefault(row["id"], row["choices"])
                if row["prediction"] is not None:
                    generated.setdefault(row["id"], []).append(row["prediction"])
                else:
                    generated.setdefault(row["id"], [])
                if settings.candidate_samples:
                    for name, answers in (row.get("sampled_answers") or {}).items():
                        sampled.setdefault(row["id"], []).extend(a for a in answers if a is not None)
    if not generated:
        raise RuntimeError("candidate pool is empty: no source records found for "
                           f"{len(wanted)} expected path(s). The scoring run must use the same flags as "
                           f"the generation run, or the digests will not match. First expected:\n"
                           f"  {wanted[0] if wanted else '-'}")

    # Labels come first so every one keeps its option text; a generated answer outside the set is appended.
    # A closed-label task with no per-example options (true/false) still gets its whole label set.
    fallback = label_space(task) or ()
    pool: Dict[str, List[Candidate]] = {}
    for example_id, answers in generated.items():
        row: List[Candidate] = []
        choices = options.get(example_id) or {label: None for label in fallback}
        for label, text in sorted(choices.items()):
            row.append(Candidate(label, label, text, "labels"))
        for source, group in (("generated", answers), ("resampled", sampled.get(example_id, []))):
            for answer in group:
                existing = next((c for c in row if task.equivalent(c.answer, answer)), None)
                if existing is None:
                    row.append(Candidate(answer, answer, None, source))
                elif source not in existing.source:
                    existing.source = f"{existing.source}+{source}"
        pool[example_id] = row
    mean = sum(len(v) for v in pool.values()) / max(len(pool), 1)
    print(f"    candidates: read {len(seen)}/{len(wanted)} source file(s), mean {mean:.2f} per example"
          + (" (resamples included)" if settings.candidate_samples else ""), flush=True)
    return pool


def add_candidate_scores(llm, task, examples, records, settings: InferenceSettings,
                         store: Path = STORE) -> None:
    """Attach this model's token log-probs for every candidate, in place, one entry per record."""
    pool = candidate_pool(task, settings, store)
    missing = [record["id"] for record in records if record["id"] not in pool]
    print(f"    candidates: pool covers {len(records) - len(missing)}/{len(records)} example(s), "
          f"mean {sum(len(pool.get(r['id'], ())) for r in records) / max(len(records), 1):.1f} per example",
          flush=True)
    rows = [pool.get(record["id"], []) for record in records]
    starts = [_answer_start(task, record) for record in records]
    scorer = CandidateScorer(llm, debias=settings.debias)
    for record, entry in zip(records, scorer.score(task, examples, [r["response"] for r in records],
                                                   starts, rows)):
        record["candidate_scores"] = entry


def _answer_start(task, record: Dict[str, Any]) -> Optional[int]:
    """Where this model's own answer begins in its response: the fixed boundary for reasoned scoring."""
    answer = task.extract_answer(record["response"]) if record.get("response") else None
    return answer.start if answer else None


def split_count(value: str) -> int:
    """CLI split size: `all` for the whole split, `none` to skip it, otherwise a count."""
    text = str(value).strip().lower()
    return -1 if text == "all" else 0 if text == "none" else int(text)


def _count(value: Optional[int]) -> Optional[int]:
    return None if value in (0, -1) else value


def records_path(settings: InferenceSettings, split: str, store: Path = STORE) -> Path:
    return inference_dir(settings, store) / f"{split}.jsonl"


def load_records(settings: InferenceSettings, split: str, store: Path = STORE) -> List[Dict[str, Any]]:
    path = records_path(settings, split, store)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def voter_settings(task: str, models: Sequence[str], voters: int, temperature: float = 0.7,
                   **overrides) -> List[InferenceSettings]:
    """One request per model and voter index, the shape a voting experiment asks the store for."""
    return [InferenceSettings(task=task, model=model, answer_temperature=temperature, voter=voter, **overrides)
            for model in models for voter in range(voters)]


def describe(wanted: Sequence[InferenceSettings]) -> str:
    return "\n".join(f"  {s.filled().task}/{s.filled().tag}--{s.digest}" for s in wanted)
