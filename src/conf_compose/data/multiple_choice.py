"""Multiple-choice reasoning tasks with lettered options: CommonsenseQA, TruthfulQA, GPQA."""

from __future__ import annotations

import random
import re
import string
from typing import Any, Dict, List, Optional, Sequence

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


class TruthfulQA(MultipleChoiceTask):
    """MC1: one true answer among several plausible misconceptions, so errors correlate across models."""
    name = "truthfulqa"
    hf_path = "truthfulqa/truthful_qa"
    hf_config = "multiple_choice"
    hf_splits = {"validation": "validation", "test": "validation"}
    letters = string.ascii_uppercase

    def to_example(self, row: Dict[str, Any], default_id: str) -> Example:
        targets = row["mc1_targets"]
        return _choice_example(self.letters, default_id, row["question"], targets["choices"],
                               [i for i, label in enumerate(targets["labels"]) if label == 1][0])


class GPQADiamond(MultipleChoiceTask):
    """Graduate-level science questions that experts get right and non-experts do not."""
    name = "gpqa"
    hf_path = "Idavidrein/gpqa"
    hf_config = "gpqa_diamond"
    hf_splits = {"validation": "train", "test": "train"}
    letters = "ABCD"

    def to_example(self, row: Dict[str, Any], default_id: str) -> Example:
        choices = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"],
                   row["Incorrect Answer 3"]]
        return _choice_example(self.letters, default_id, row["Question"], choices, 0)


def _choice_example(letters: str, example_id: str, question: str, choices: Sequence[str],
                    correct: int) -> Example:
    """Options are shuffled deterministically per example, since the source lists the answer first."""
    if len(choices) > len(letters):
        raise ValueError(f"{example_id}: {len(choices)} options but only {len(letters)} letters available")
    order = list(range(len(choices)))
    random.Random(example_id).shuffle(order)
    used = letters[:len(choices)]
    text = "\n".join(f"{used[position]}) {choices[index]}" for position, index in enumerate(order))
    answer = used[order.index(correct)]
    return Example(id=example_id, question=f"{question}\n{text}", answer=answer,
                   meta={"choices": {used[p]: choices[i] for p, i in enumerate(order)}, "options": list(used)})
