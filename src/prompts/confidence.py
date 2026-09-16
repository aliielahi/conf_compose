"""Prompts and content-free inputs used by the confidence estimators."""

from .base import Template


class VerbalizedPrompts:
    rate = Template(
        "Rate your confidence in your final answer on a scale from 0 to 10, where 0 means you are guessing "
        "randomly, 5 means you are somewhat confident but unsure, and 10 means you are almost certain it is "
        "correct. Only output one integer in the [0, 10] range."
    )


class ContentFreeInputs:
    inputs = ("N/A", "", "[MASK]")
