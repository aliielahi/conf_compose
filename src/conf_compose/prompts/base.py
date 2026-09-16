"""Prompt template primitive: a format string with named fields."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    text: str

    def __call__(self, **fields: str) -> str:
        return self.text.format(**fields)
