"""End-to-end pipelines that combine models, tasks and confidence estimators."""

from .zero_shot import ZeroShotConfig, run_zero_shot, signal, signal_names

__all__ = ["ZeroShotConfig", "run_zero_shot", "signal", "signal_names"]
