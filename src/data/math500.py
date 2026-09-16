"""MATH-500: competition math with LaTeX answers in \\boxed{}; graded with math-verify when installed."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .base import Answer, Example, Task


def last_boxed(response: str) -> Optional[Answer]:
    start = response.rfind("\\boxed{")
    if start < 0:
        return None
    i = content_start = start + len("\\boxed{")
    depth = 1
    while i < len(response):
        if response[i] == "{":
            depth += 1
        elif response[i] == "}":
            depth -= 1
            if depth == 0:
                return Answer(response[content_start:i], content_start, i)
        i += 1
    return None


def normalize_latex(text: str) -> str:
    text = re.sub(r"\\(left|right|!|,|;|quad)", "", text)
    text = re.sub(r"\\text\{(.*?)\}", r"\1", text)
    text = text.replace("dfrac", "frac").replace("tfrac", "frac").replace("^\\circ", "").replace("$", "")
    return re.sub(r"\s+", "", text).rstrip(".")


class MATH500(Task):
    name = "math500"
    hf_path = "HuggingFaceH4/MATH-500"
    validation_source = "test"
    val_size = 100
    instruction = "Solve the problem step by step. Put your final answer within \\boxed{}."
    answer_prefix = "The final answer is \\boxed{"

    def to_example(self, row: Dict[str, Any], default_id: str) -> Example:
        return Example(id=row["unique_id"], question=row["problem"], answer=row["answer"],
                       meta={"subject": row["subject"], "level": row["level"], "solution": row["solution"]})

    def extract_answer(self, response: str) -> Optional[Answer]:
        return last_boxed(response)

    def is_correct(self, predicted: Optional[str], example: Example) -> bool:
        if predicted is None:
            return False
        try:
            from math_verify import parse, verify
            return bool(verify(parse(f"${example.answer}$"), parse(f"${predicted}$")))
        except ImportError:
            return normalize_latex(predicted) == normalize_latex(example.answer)
