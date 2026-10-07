"""What defines one debate cell: task, group and everything that changes what the agents write."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from conf_compose.constants import RESULTS_DIR

DEBATE_STORE = RESULTS_DIR / "debate_inferences"
# Bump when the revision prompt changes, so old and new turns never share a cell.
PROMPT_VERSION = 1


@dataclass(frozen=True)
class DebateSettings:
    """Round 0 comes from the inference store; rounds >= 1 are generated under these settings."""
    task: str
    group: Tuple[str, ...]
    max_tokens: int
    voter: int = 0
    match: str = "_cs7s"
    temperature: float = 0.7
    context: int = 32768
    limit: Optional[int] = None
    seed: int = 0
    prompt_version: int = PROMPT_VERSION

    @property
    def decoding(self) -> str:
        return f"s{int(round(self.temperature * 100)):02d}v{self.voter}"

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:8]

    @property
    def name(self) -> str:
        size = f"_n{self.limit}" if self.limit else ""
        return f"{'+'.join(self.group)}--{self.decoding}{size}--{self.digest}"

    def directory(self, out: Path = DEBATE_STORE) -> Path:
        return Path(out) / self.task / self.name

    def round_path(self, model: str, round_index: int, out: Path = DEBATE_STORE) -> Path:
        return self.directory(out) / f"round_{round_index}" / f"{model}.jsonl"

    def scores_path(self, model: str, through: int, out: Path = DEBATE_STORE) -> Path:
        """Candidate scores for rounds 0..through, over the pool of every answer proposed up to that round."""
        return self.directory(out) / f"candidates_through_round_{through}" / f"{model}.jsonl"

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "digest": self.digest, "name": self.name}
