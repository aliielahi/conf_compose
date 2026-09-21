"""The shared inference store: one model's answers and confidence signals, generated once and reused."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from conf_compose.constants import RESULTS_DIR, SAMPLING, TASKS

from conf_compose.utils.llm_calls import LLM

from .runs import split_summary
from .zero_shot import RECORD_SCHEMA, ZeroShotConfig, run_zero_shot

STORE = RESULTS_DIR / "inferences"


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
        return f"{self.model.replace('/', '__')}--{self.decoding}_k{self.consistency_samples}_{self.size}"

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self.filled()), sort_keys=True).encode()).hexdigest()[:8]

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
                            consistency_temperatures=(settings.consistency_temperature,))
    llm.execution = f"{settings.task}:{settings.decoding}"
    validation = [] if settings.n_val == 0 else task.load("validation", n=_count(settings.n_val))
    test = [] if settings.n_test == 0 else task.load("test", n=_count(settings.n_test))

    start, timings = time.time(), {}
    combined = run_zero_shot(llm, task, validation + test, config, timings)
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
