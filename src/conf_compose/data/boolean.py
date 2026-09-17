"""Boolean reasoning tasks with true/false answers: BoolQ and ProntoQA."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from conf_compose.prompts import BooleanReasoning

from .base import Answer, Example, Task

PRONTOQA_URL = "https://raw.githubusercontent.com/teacherpeterpan/Logic-LLM/main/data/ProntoQA/dev.json"

_WORDS = r"true|false|yes|no"
_CUE = re.compile(rf"answer is[:\s]*\**\s*({_WORDS})\b", re.IGNORECASE)
_ANY = re.compile(rf"\b({_WORDS})\b", re.IGNORECASE)


class BooleanTask(Task):
    prompts = BooleanReasoning

    def extract_answer(self, response: str) -> Optional[Answer]:
        matches = list(_CUE.finditer(response))
        if matches:
            m = matches[-1]
            return Answer(m.group(1).lower(), m.start(1), m.end(1))
        matches = list(_ANY.finditer(response))
        if not matches:
            return None
        m = matches[-1]
        return Answer(m.group(1).lower(), m.start(1), m.end(1), explicit=False)

    def equivalent(self, a: str, b: str) -> bool:
        return _normalize(a) == _normalize(b)


class BoolQ(BooleanTask):
    name = "boolq"
    hf_path = "google/boolq"
    hf_splits = {"train": "train", "validation": "train", "test": "validation"}

    def to_example(self, row: Dict[str, Any], default_id: str) -> Example:
        question = f"Passage: {row['passage']}\n\nStatement: {row['question']}\nIs the statement true or false?"
        return Example(id=default_id, question=question, answer=_normalize(str(row["answer"])))


class ProntoQA(BooleanTask):
    name = "prontoqa"
    hf_path = "json"
    hf_splits = {"validation": "train", "test": "train"}
    hf_kwargs = {"data_files": PRONTOQA_URL}

    def to_example(self, row: Dict[str, Any], default_id: str) -> Example:
        options = {option.split(")")[0].strip().upper(): option.split(")", 1)[1].strip()
                   for option in row["options"]}
        question = f"{row['context']}\n\n{row['question']}"
        return Example(id=row["id"], question=question, answer=_normalize(options[row["answer"].strip().upper()]),
                       meta={"explanation": row["explanation"]})


def _normalize(value: str) -> str:
    value = value.strip().lower()
    return "true" if value in ("true", "yes") else "false" if value in ("false", "no") else value
