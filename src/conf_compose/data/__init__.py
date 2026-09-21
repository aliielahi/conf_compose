"""Reasoning datasets behind one Task interface: load, prompt, extract answer, grade."""

from .base import Answer, Example, Task
from .boolean import BoolQ, ProntoQA
from .math500 import MATH500
from .multiple_choice import CommonsenseQA, GPQADiamond, TruthfulQA
from .numeric import GSM8K, SVAMP

TASKS = {task.name: task for task in (GSM8K, SVAMP, MATH500, CommonsenseQA, TruthfulQA,
                                     GPQADiamond, BoolQ, ProntoQA)}


def get_task(name: str) -> Task:
    try:
        return TASKS[name.lower()]()
    except KeyError:
        raise ValueError(f"unknown task {name!r}; available: {sorted(TASKS)}") from None


__all__ = ["Answer", "Example", "Task", "GSM8K", "SVAMP", "MATH500", "CommonsenseQA", "TruthfulQA",
           "GPQADiamond", "BoolQ", "ProntoQA", "TASKS", "get_task"]
