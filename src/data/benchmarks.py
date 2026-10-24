"""Math benchmarks with gold final answers: GSM8K test and MATH-500."""

from datasets import load_dataset

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def load(name):
    """Rows with id, problem and gold answer."""
    if name == "gsm8k":
        rows = load_dataset("openai/gsm8k", "main", split="test")
        return [{"id": f"gsm8k-{i}", "problem": r["question"], "answer": r["answer"].split("####")[-1].strip()}
                for i, r in enumerate(rows)]
    if name == "math500":
        rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]} for r in rows]
    raise ValueError(name)
