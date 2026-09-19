"""End-to-end pipelines that combine models, tasks and confidence estimators."""

from .confidence import ConfidenceConfig, ConfidenceOutput, estimate_confidence
from .zero_shot import ZeroShotConfig, run_zero_shot, signal, signal_names

__all__ = ["ConfidenceConfig", "ConfidenceOutput", "estimate_confidence", "ZeroShotConfig", "run_zero_shot",
           "signal", "signal_names"]
