"""Example: collect CAMEL conversation logs with an OpenAI GPT backbone.

This script is intentionally a sample data-collection pipeline. MATU itself
only requires already-collected conversation logs in the JSON format written by
this script.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from matu.constants import DEFAULT_EMBEDDING_MODEL
from matu.env import load_dotenv


def require_camel():
    try:
        from camel.configs import ChatGPTConfig
        from camel.models import ModelFactory
        from camel.societies import RolePlaying
        from camel.types import ModelPlatformType, ModelType
    except ImportError as exc:
        raise SystemExit("Please install CAMEL first, e.g. `pip install camel-ai`.") from exc
    return ChatGPTConfig, ModelFactory, RolePlaying, ModelPlatformType, ModelType


def model_type_from_name(model_type_enum: Any, name: str) -> Any:
    normalized = name.upper().replace("-", "_").replace(".", "_")
    aliases = {
        "GPT_4O": "GPT_4O",
        "GPT_4O_MINI": "GPT_4O_MINI",
        "GPT_5": "GPT_5",
        "GPT_5_MINI": "GPT_5_MINI",
    }
    attr = aliases.get(normalized, normalized)
    if not hasattr(model_type_enum, attr):
        raise ValueError(f"Unsupported CAMEL ModelType: {name}. Add it to model_type_from_name if needed.")
    return getattr(model_type_enum, attr)


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


def run_conversation(society: Any, round_limit: int) -> list[dict[str, str]]:
    log: list[dict[str, str]] = []
    input_msg = society.init_chat()

    for _ in range(round_limit):
        assistant_response, user_response = society.step(input_msg)
        if getattr(assistant_response, "terminated", False) or getattr(user_response, "terminated", False):
            break

        log.append({"role": "user", "output": user_response.msg.content})
        log.append({"role": "assistant", "output": assistant_response.msg.content})

        if "CAMEL_TASK_DONE" in user_response.msg.content:
            break
        input_msg = assistant_response.msg

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
    parser = argparse.ArgumentParser(description="Collect CAMEL GPT conversation logs on a local MATH dataset.")
    parser.add_argument("--data_root", type=Path, default=Path("MATH/test"))
    parser.add_argument("--out", type=Path, default=Path("outputs/conversation_logs_camel_gpt.json"))
    parser.add_argument("--category", default=None, help="Optional MATH category, e.g. algebra.")
    parser.add_argument("--max_items", type=int, default=10)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--round_limit", type=int, default=10)
    parser.add_argument("--model", default="GPT_4O", help="CAMEL ModelType name, e.g. GPT_4O or GPT_5_MINI.")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding_out_dir", type=Path, default=None, help="If set, also save user/assistant embedding matrices.")
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Please set OPENAI_API_KEY in your environment or local .env file.")

    ChatGPTConfig, ModelFactory, RolePlaying, ModelPlatformType, ModelType = require_camel()
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=model_type_from_name(ModelType, args.model),
        model_config_dict=ChatGPTConfig(temperature=args.temperature).as_dict(),
    )

    items = load_math_items(args.data_root, args.max_items, args.seed, args.category)
    results: dict[str, list[list[dict[str, str]]]] = {}

    for key, data in tqdm(items, desc="Problems"):
        problem_text = str(data["problem"]).replace("{", "{{").replace("}", "}}")
        task_prompt = f'Answer the following math problem: "{problem_text}".'
        task_kwargs = {
            "task_prompt": task_prompt,
            "with_task_specify": True,
            "task_specify_agent_kwargs": {"model": model},
        }
        user_role_kwargs = {
            "user_role_name": "The Curious Math Enthusiast.",
            "user_agent_kwargs": {"model": model},
        }
        assistant_role_kwargs = {
            "assistant_role_name": "The Master Mathematician.",
            "assistant_agent_kwargs": {"model": model},
        }

        runs = []
        for _ in range(args.runs):
            society = RolePlaying(**task_kwargs, **user_role_kwargs, **assistant_role_kwargs)
            runs.append(run_conversation(society, args.round_limit))
        results[key] = runs

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved logs to {args.out}")

    if args.embedding_out_dir is not None:
        embed_and_save(results, args.embedding_out_dir, args.embedding_model)


if __name__ == "__main__":
    main()
