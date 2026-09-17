"""Shared task interface: examples, prompt construction, answer extraction and grading."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from conf_compose.constants import SEED, TASKS


@dataclass
class Example:
    id: str
    question: str
    answer: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    text: str
    start: int
    end: int
    explicit: bool = True


class Task:
    name: str = ""
    hf_path: str = ""
    hf_config: Optional[str] = None
    hf_splits: Dict[str, str] = {"train": "train", "validation": "train", "test": "test"}
    hf_kwargs: Dict[str, Any] = {}
    prompts: Any = None

    def load(self, split: str = "test", n: Optional[int] = None, seed: int = SEED) -> List[Example]:
        if split not in self.hf_splits:
            raise ValueError(f"unknown split {split!r}")
        source = self.hf_splits[split]
        examples = [self.to_example(row, f"{self.name}-{source}-{i}") for i, row in self._rows(split, seed)]
        if n is not None and n < len(examples):
            examples = random.Random(seed).sample(examples, n)
        return examples

    def _rows(self, split: str, seed: int):
        from datasets import load_dataset

        rows = list(enumerate(load_dataset(self.hf_path, self.hf_config, split=self.hf_splits[split],
                                           **self.hf_kwargs)))
        if self.hf_splits["validation"] != self.hf_splits["test"] or split == "train":
            return rows
        pool_size = min(TASKS[self.name]["val_pool_size"], len(rows))
        held_out = set(random.Random(seed).sample(range(len(rows)), pool_size))
        return [row for row in rows if (row[0] in held_out) == (split == "validation")]

    @property
    def answer_prefix(self) -> str:
        return self.prompts.answer_prefix

    def prompt(self, example: Example) -> str:
        return self.prompts.solve(question=example.question, word_limit=TASKS[self.name]["word_limit"])

    def null_prompt(self, null_input: str) -> str:
        return self.prompts.solve(question=null_input, word_limit=TASKS[self.name]["word_limit"])

    def to_example(self, row: Dict[str, Any], default_id: str) -> Example:
        raise NotImplementedError

    def extract_answer(self, response: str) -> Optional[Answer]:
        raise NotImplementedError

    def equivalent(self, a: str, b: str) -> bool:
        raise NotImplementedError

    def is_correct(self, predicted: Optional[str], example: Example) -> bool:
        return predicted is not None and self.equivalent(predicted, example.answer)
