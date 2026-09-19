"""Debate generation: protocol settings, the synchronous engine, and reusable traces."""

from .confidence import add_confidence, final_answers
from .engine import run_debate
from .protocol import DebateConfig
from .trace import ExampleTrace, Turn, append_traces, completed_ids, read_traces, turn_id

__all__ = ["run_debate", "add_confidence", "final_answers", "DebateConfig", "ExampleTrace", "Turn", "append_traces", "completed_ids", "read_traces",
           "turn_id"]
