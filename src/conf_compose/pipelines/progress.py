"""Run status: a percentage line in the run log, and one always-current snapshot at the repo root."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from conf_compose.constants import ROOT

SNAPSHOT = ROOT / "current_run.log"


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


class Status:
    """One run's progress, echoed to stdout and mirrored to `current_run.log` after every update."""

    def __init__(self, name: str, total: int = 0, snapshot: Optional[Path] = None, keep: int = 8):
        self.name = name
        self.total = total
        self.done = 0
        self.stage_text = "starting"
        self.started = time.time()
        self.events: List[str] = []
        self.keep = keep
        self.snapshot = Path(snapshot) if snapshot else SNAPSHOT
        self.write()

    @property
    def percent(self) -> float:
        return 100.0 * self.done / self.total if self.total else 0.0

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    def stage(self, text: str) -> None:
        self.stage_text = text
        self._log(text)

    def step(self, note: str = "", done: int = 1) -> None:
        self.done += done
        self._log(f"[{self.percent:5.1f}%] {self.done}/{self.total} {note}".rstrip())

    def finish(self, text: str = "done") -> None:
        self.stage_text = text
        self._log(f"{text} in {_clock(self.elapsed)}")
        self.write(final=True)

    def _log(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {text}"
        print(line, flush=True)
        self.events.append(line)
        self.events = self.events[-self.keep:]
        self.write()

    def write(self, final: bool = False) -> None:
        """Rewritten in full each time, so the file is a snapshot and never needs scrolling."""
        eta = ""
        if self.done and self.total and not final:
            eta = f" | eta {_clock(self.elapsed / self.done * (self.total - self.done))}"
        bar = ""
        if self.total:
            filled = int(round(self.percent / 5))
            bar = f"\n[{'#' * filled}{'.' * (20 - filled)}] {self.percent:.1f}%  {self.done}/{self.total}{eta}"
        body = (f"run     : {self.name}\n"
                f"started : {datetime.fromtimestamp(self.started):%Y-%m-%d %H:%M:%S}  "
                f"(elapsed {_clock(self.elapsed)})\n"
                f"stage   : {self.stage_text}{bar}\n\nrecent:\n" + "\n".join(f"  {e}" for e in self.events) + "\n")
        try:
            temporary = self.snapshot.with_suffix(".tmp")
            temporary.write_text(body)
            temporary.replace(self.snapshot)
        except OSError:
            pass
