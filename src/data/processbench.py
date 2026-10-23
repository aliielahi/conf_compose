"""ProcessBench: math solutions split into steps, labelled with the index of the first wrong step (-1 if none)."""

from datasets import load_dataset

SPLITS = ("gsm8k", "math", "olympiadbench", "omnimath")


def load(split):
    """Rows with id, problem, steps (list of str) and label."""
    return [{k: row[k] for k in ("id", "problem", "steps", "label")}
            for row in load_dataset("Qwen/ProcessBench", split=split)]
