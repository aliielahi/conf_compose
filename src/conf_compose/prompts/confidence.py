"""Prompts and content-free inputs used by the confidence estimators."""

from .base import Template


class VerbalizedPrompts:
    rate = Template(
        "Rate your confidence that the answer {answer} is correct, on a scale from 0 to 10, where 0 means you are "
        "guessing randomly, 5 means you are somewhat confident but unsure, and 10 means you are almost certain it is "
        "correct. Only output one integer in the [0, 10] range."
    )


class VerificationPrompts:
    check = Template(
        "Question: {question}\n\nProposed solution:\n{response}\n\nProposed final answer: {answer}\n\n"
        "Is the proposed final answer correct? Answer with only True or False."
    )
    labels = ("True", "False")


class ContentFreeInputs:
    inputs = ("N/A", "", "[MASK]")
