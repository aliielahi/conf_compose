"""Shared task interface: examples, prompt construction, answer extraction and grading."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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


class Task:
    name: str = ""
    hf_path: str = ""
    hf_config: Optional[str] = None
    validation_source: str = "train"
    val_size: int = 1000
    instruction: str = ""
    answer_prefix: str = ""

    def load(self, split: str = "test", n: Optional[int] = None, seed: int = 42) -> List[Example]:
        if split not in ("train", "validation", "test"):
            raise ValueError(f"unknown split {split!r}")
        source = self.validation_source if split == "validation" else split
        examples = [self.to_example(row, f"{self.name}-{source}-{i}") for i, row in self._rows(split, seed)]
        if n is not None and n < len(examples):
            examples = random.Random(seed).sample(examples, n)
        return examples

    def _rows(self, split: str, seed: int):
        from datasets import load_dataset

        source = self.validation_source if split == "validation" else split
        rows = list(enumerate(load_dataset(self.hf_path, self.hf_config, split=source)))
        if split == "train" or (split == "test" and self.validation_source == "train"):
            return rows
        held_out = set(random.Random(seed).sample(range(len(rows)), min(self.val_size, len(rows))))
        return [row for row in rows if (row[0] in held_out) == (split == "validation")]

    def prompt(self, example: Example) -> str:
        return self.format(example.question)

    def null_prompt(self, null_input: str = "N/A") -> str:
        return self.format(null_input)

    def format(self, question: str) -> str:
        return f"{question}\n\n{self.instruction}"

    def to_example(self, row: Dict[str, Any], default_id: str) -> Example:
        raise NotImplementedError

    def extract_answer(self, response: str) -> Optional[Answer]:
        raise NotImplementedError

    def is_correct(self, predicted: Optional[str], example: Example) -> bool:
        raise NotImplementedError
