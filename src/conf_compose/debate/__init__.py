"""Debate generation: protocol settings, the synchronous engine, and reusable traces."""

from .analysis import flips, label_suspects
from .confidence import add_confidence, final_answers
from .engine import run_debate
from .protocol import DebateConfig
from .trace import ExampleTrace, Turn, append_traces, completed_ids, read_traces, turn_id

__all__ = ["run_debate", "add_confidence", "final_answers", "flips", "label_suspects", "DebateConfig", "ExampleTrace", "Turn", "append_traces", "completed_ids", "read_traces",
           "turn_id"]
