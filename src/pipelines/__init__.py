"""End-to-end pipelines that combine models, tasks and confidence estimators."""

from .zero_shot import run_zero_shot, signal, signal_names

__all__ = ["run_zero_shot", "signal", "signal_names"]
