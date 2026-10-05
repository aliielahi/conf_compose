"""Example: collect two-role conversation logs with a local/HF Qwen model.

This is a lightweight sample for users who prefer open-source backbones. It
does not depend on CAMEL; it simply alternates a user role and an assistant role
with separate system prompts.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from matu.constants import DEFAULT_EMBEDDING_MODEL


ASSISTANT_SYSTEM = (
    "You are The Master Mathematician. Be precise, show clear reasoning, "
    "and end with a line `FINAL ANSWER: <value>` when ready."
)

USER_SYSTEM = (
    "You are The Curious Math Enthusiast. Ask concise follow-ups or request "
    "clarification. If the final answer is clearly correct, reply exactly "
    "`CAMEL_TASK_DONE`."
)


class HFChatModel:
    def __init__(self, model_name: str, cache_dir: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            cache_dir=cache_dir,
        )

    @torch.inference_mode()
    def generate(self, prompt: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return text.split("\nUser:")[0].split("\nAssistant:")[0].strip()


def load_math_items(data_root: Path, max_items: int, seed: int, category: str | None) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    categories = [category] if category else sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    for cat in categories:
        cat_dir = data_root / cat
        if not cat_dir.exists():
            continue
        for path in sorted(cat_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            items.append((f"{cat}_{path.stem}", data))

    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:max_items]


def next_turn(
    hf: HFChatModel,
    system_prompt: str,
    transcript: list[tuple[str, str]],
    next_role: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    lines = [system_prompt, ""]
    for role, text in transcript:
        lines.append(f"{role}: {text}")
    lines.append(f"{next_role}:")
    return hf.generate("\n".join(lines), temperature, top_p, max_new_tokens)


def run_conversation(
    hf: HFChatModel,
    problem: str,
    round_limit: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    log: list[dict[str, str]] = []
    transcript: list[tuple[str, str]] = []

    user_text = (
        "Solve the math problem step by step, then print `FINAL ANSWER: <value>` "
        f"on the last line.\nProblem: {problem}"
    )
    transcript.append(("User", user_text))
    log.append({"role": "user", "output": user_text})

    assistant_text = next_turn(hf, ASSISTANT_SYSTEM, transcript, "Assistant", temperature, top_p, max_new_tokens)
    log.append({"role": "assistant", "output": assistant_text})

    for _ in range(round_limit - 1):
        transcript.append(("Assistant", assistant_text))
        user_text = next_turn(hf, USER_SYSTEM, transcript, "User", temperature, top_p, max_new_tokens)
        if not user_text:
            user_text = "Please clarify your last step."
        log.append({"role": "user", "output": user_text})
        if "CAMEL_TASK_DONE" in user_text:
            break

        transcript.append(("User", user_text))
        assistant_text = next_turn(hf, ASSISTANT_SYSTEM, transcript, "Assistant", temperature, top_p, max_new_tokens)
        log.append({"role": "assistant", "output": assistant_text})
        if "FINAL ANSWER" in assistant_text.upper():
            break

    return log


def embed_and_save(logs: dict[str, list[list[dict[str, str]]]], out_dir: Path, model_name: str) -> None:
    model = SentenceTransformer(model_name, model_kwargs={"device_map": "auto"}, tokenizer_kwargs={"padding_side": "left"})
    out_dir.mkdir(parents=True, exist_ok=True)
    user_dict = {}
    assistant_dict = {}

    for key, runs in logs.items():
        user_runs = []
        assistant_runs = []
        for run in runs:
            user_texts = [turn["output"] for turn in run if turn.get("role") == "user"]
            assistant_texts = [turn["output"] for turn in run if turn.get("role") == "assistant"]
            user_runs.append(np.asarray(model.encode(user_texts, normalize_embeddings=True), dtype=np.float32) if user_texts else np.array([]))
            assistant_runs.append(np.asarray(model.encode(assistant_texts, normalize_embeddings=True), dtype=np.float32) if assistant_texts else np.array([]))
        user_dict[key] = user_runs
        assistant_dict[key] = assistant_runs

    with (out_dir / "user_embedding_matrices.pkl").open("wb") as f:
        pickle.dump(user_dict, f, protocol=4)
    with (out_dir / "assistant_embedding_matrices.pkl").open("wb") as f:
        pickle.dump(assistant_dict, f, protocol=4)
    metadata = {
        "embedding_model": model_name,
        "embedding_dim": int(model.get_sentence_embedding_dimension() or 0),
        "roles": ["user", "assistant"],
    }
    with (out_dir / "embedding_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved embeddings to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect HF/Qwen two-role conversation logs on local MATH.")
    parser.add_argument("--data_root", type=Path, default=Path("MATH/test"))
    parser.add_argument("--out", type=Path, default=Path("outputs/conversation_logs_hf_qwen.json"))
    parser.add_argument("--category", default=None)
    parser.add_argument("--max_items", type=int, default=10)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--round_limit", type=int, default=10)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding_out_dir", type=Path, default=None, help="If set, also save user/assistant embedding matrices.")
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    hf = HFChatModel(args.model, args.cache_dir)
    items = load_math_items(args.data_root, args.max_items, args.seed, args.category)
    results: dict[str, list[list[dict[str, str]]]] = {}

    for key, data in tqdm(items, desc="Problems"):
        runs = []
        for _ in range(args.runs):
            runs.append(
                run_conversation(
                    hf,
                    str(data["problem"]),
                    args.round_limit,
                    args.temperature,
                    args.top_p,
                    args.max_new_tokens,
                )
            )
        results[key] = runs

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved logs to {args.out}")

    if args.embedding_out_dir is not None:
        embed_and_save(results, args.embedding_out_dir, args.embedding_model)


if __name__ == "__main__":
    main()
