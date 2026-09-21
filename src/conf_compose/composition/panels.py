"""Evidence panels: which sources vote, across model set, estimator family and sample budget."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .candidates import Support, support_of
from .evidence import Item, Stream

# Saved signal families; each rates only its own model's answer, so it needs the target to match.
SIGNALS = {"verification": "verification", "verbalized": "verbalized", "seq": "seq_response",
           "consistency_full": "consistency_t0.7", "ver_target": "verification_target"}
# Families that rate a shared target by construction, so they stay eligible when it is not the model's answer.
SHARED_TARGET = ("ver_target",)
PROBABILITY_SIGNALS = ("verification", "verbalized", "consistency_full")
SHORT = {"consistency": "cons", "verification": "ver", "verbalized": "verb", "seq": "seq",
         "ver_target": "vertgt"}


@dataclass
class Source:
    """One scalar rating of a fixed target, from one model and one estimator family."""
    name: str
    model: str
    family: str
    score: Optional[float]
    eligible: bool = True


@dataclass
class Panel:
    """One experiment cell: sample sub-streams per model, plus per-model signal families."""
    models: Sequence[str]
    streams_per_model: int = 1
    samples_per_stream: int = 5
    families: Sequence[str] = ("consistency",)
    label: str = ""

    @property
    def kind(self) -> str:
        """`voters` are separately generated runs; `split` carves one run's samples into sub-streams."""
        return "split" if len(self.models) == 1 and self.streams_per_model > 1 else "voters"

    @property
    def distinct(self) -> int:
        return len({base_model(model) for model in self.models})

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        families = "+".join(SHORT.get(f, f) for f in self.families)
        return f"{self.kind}_m{len(self.models)}_s{self.streams_per_model}_k{self.samples_per_stream}_{families}"

    @property
    def sample_calls(self) -> int:
        """Sampled generations consumed, the comparable budget unit across panels."""
        if "consistency" not in self.families:
            return 0
        return len(self.models) * self.streams_per_model * self.samples_per_stream

    @property
    def answer_calls(self) -> int:
        """Answer generations consumed; a split panel reuses one answer, separate voters need one each."""
        return len(self.models)

    @property
    def signal_calls(self) -> int:
        return len(self.models) * len([f for f in self.families if f != "consistency"])


def base_model(model: str) -> str:
    """Model tag with decoding, voter and digest stripped, so two runs of one model count as one model."""
    model = model.split("--")[0]
    for marker in ("_seed", "_rep", "_run"):
        if marker in model:
            return model.split(marker)[0]
    return model


def blocks(samples: Sequence[Optional[str]], streams: int, per_stream: int) -> List[Sequence[Optional[str]]]:
    """Disjoint consecutive sample blocks, one per sub-stream; empty when the budget does not fit."""
    if streams * per_stream > len(samples):
        return []
    return [samples[i * per_stream:(i + 1) * per_stream] for i in range(streams)]


def sources(task, item: Item, target: str, candidates: Sequence[str], panel: Panel,
            streams: Optional[Sequence[Stream]] = None, variant: str = "add_half") -> List[Source]:
    """Every source's rating of the same fixed target; ineligible self-ratings are kept but flagged."""
    by_model = {stream.model: stream for stream in (streams if streams is not None else item.streams)}
    out: List[Source] = []
    for model in panel.models:
        stream = by_model.get(model)
        if stream is None:
            return []
        own = stream.answer is not None and task.equivalent(stream.answer, target)
        if "consistency" in panel.families:
            sample_blocks = blocks(stream.samples, panel.streams_per_model, panel.samples_per_stream)
            if not sample_blocks:
                return []
            for index, block in enumerate(sample_blocks):
                score = support_of(task, block, candidates).binary(target, variant)
                out.append(Source(f"{model}:consistency:{index}", model, "consistency", score))
        for family in panel.families:
            if family == "consistency":
                continue
            score = stream.signals.get(SIGNALS[family])
            eligible = own or family in SHARED_TARGET
            out.append(Source(f"{model}:{family}", model, family, score, eligible=eligible))
    return out


def available(sources_: Sequence[Source], require_eligible: bool = True) -> List[Source]:
    return [s for s in sources_ if s.score is not None and (s.eligible or not require_eligible)]


def model_panels(models: Sequence[str], sizes: Sequence[int], samples: Sequence[int],
                 families: Sequence[Sequence[str]]) -> List[Panel]:
    """Matched grid: heterogeneous panels of S models against homogeneous panels of S sub-streams."""
    panels: List[Panel] = []
    for size in sizes:
        for per_stream in samples:
            for family in families:
                if len(models) >= size:
                    panels.append(Panel(tuple(models[:size]), 1, per_stream, tuple(family)))
                for model in models:
                    panels.append(Panel((model,), size, per_stream, tuple(family),
                                        label=f"homo[{model}]_s{size}_k{per_stream}_{'+'.join(family)}"))
    return panels
