"""Multiple-choice reasoning tasks with lettered options: CommonsenseQA."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from conf_compose.prompts import MultipleChoiceReasoning

from .base import Answer, Example, Task

_CUE = re.compile(r"answer is[:\s]*\**\s*\(?([A-Za-z])\)?", re.IGNORECASE)


class MultipleChoiceTask(Task):
    prompts = MultipleChoiceReasoning
    letters = "ABCDE"

    def extract_answer(self, response: str) -> Optional[Answer]:
        matches = [m for m in _CUE.finditer(response) if m.group(1).upper() in self.letters]
        if matches:
            m = matches[-1]
            return Answer(m.group(1).upper(), m.start(1), m.end(1))
        for pattern in (rf"\(([{self.letters}{self.letters.lower()}])\)", rf"\b([{self.letters}])\b"):
            matches = list(re.finditer(pattern, response))
            if matches:
                m = matches[-1]
                return Answer(m.group(1).upper(), m.start(1), m.end(1), explicit=False)
        return None

    def equivalent(self, a: str, b: str) -> bool:
        return a.strip().upper() == b.strip().upper()


class CommonsenseQA(MultipleChoiceTask):
    name = "csqa"
    hf_path = "tau/commonsense_qa"
    hf_splits = {"train": "train", "validation": "train", "test": "validation"}

    def to_example(self, row: Dict[str, Any], default_id: str) -> Example:
        labels, texts = row["choices"]["label"], row["choices"]["text"]
        options = "\n".join(f"{label}) {text}" for label, text in zip(labels, texts))
        return Example(id=row["id"], question=f"{row['question']}\n{options}", answer=row["answerKey"].strip().upper(),
                       meta={"choices": dict(zip(labels, texts)), "concept": row["question_concept"]})
