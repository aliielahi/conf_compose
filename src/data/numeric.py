"""Math word problems with a single numeric answer: GSM8K and SVAMP."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from prompts import NumericReasoning

from .base import Answer, Example, Task

_NUMBER = r"-?(?:\$\s*)?\d[\d,]*(?:\.\d+)?|-?\.\d+"
_CUE = re.compile(rf"answer is[:\s]*\**\s*({_NUMBER})", re.IGNORECASE)
_ANY = re.compile(_NUMBER)


def normalize_number(text: str) -> Optional[float]:
    cleaned = re.sub(r"[\s$,]", "", text).rstrip(".")
    try:
        return float(cleaned)
    except ValueError:
        return None


class NumericTask(Task):
    prompts = NumericReasoning

    def extract_answer(self, response: str) -> Optional[Answer]:
        matches = list(_CUE.finditer(response))
        if matches:
            m = matches[-1]
            return Answer(m.group(1), m.start(1), m.end(1))
        matches = list(_ANY.finditer(response))
        if not matches:
            return None
        m = matches[-1]
        return Answer(m.group(0), m.start(), m.end())

    def equivalent(self, a: str, b: str) -> bool:
        x, y = normalize_number(a), normalize_number(b)
        return x is not None and y is not None and abs(x - y) < 1e-6


class GSM8K(NumericTask):
    name = "gsm8k"
    hf_path = "openai/gsm8k"
    hf_config = "main"

    def to_example(self, row: Dict[str, Any], default_id: str) -> Example:
        solution, _, final = row["answer"].rpartition("####")
        return Example(id=default_id, question=row["question"], answer=final.strip().replace(",", ""),
                       meta={"solution": solution.strip()})


class SVAMP(NumericTask):
    name = "svamp"
    hf_path = "ChilleD/SVAMP"
    val_size = 200

    def to_example(self, row: Dict[str, Any], default_id: str) -> Example:
        value = float(row["Answer"])
        answer = str(int(value)) if value.is_integer() else str(value)
        question = f"{row['Body'].strip()} {row['Question'].strip()}"
        return Example(id=str(row["ID"]), question=question, answer=answer,
                       meta={"equation": row["Equation"], "type": row["Type"]})
