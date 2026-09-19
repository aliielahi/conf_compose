"""Debate generation: protocol settings, the synchronous engine, and reusable traces."""

from .engine import run_debate
from .protocol import DebateConfig
from .trace import ExampleTrace, Turn, append_traces, completed_ids, read_traces, turn_id

__all__ = ["run_debate", "DebateConfig", "ExampleTrace", "Turn", "append_traces", "completed_ids", "read_traces",
           "turn_id"]
