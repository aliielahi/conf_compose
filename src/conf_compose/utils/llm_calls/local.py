"""Shared base for in-process models: chat rendering, disk cache, deduplicated continuation scoring."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .core import BaseLLM, Conversation, Generation, PromptLike, normalize_prompt
from .hf_models import HF_models


class LocalLLM(BaseLLM):
    def __init__(self, model: str, *, apply_chat_template: bool = True,
                 chat_template_kwargs: Optional[Dict[str, Any]] = None,
                 generation_prefix: str = "", **kwargs: Any):
        super().__init__(HF_models.get(model, model), **kwargs)
        self.apply_chat_template = apply_chat_template
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.generation_prefix = generation_prefix
        self.tokenizer: Any = None

    def render(self, system: Optional[str], messages: Conversation) -> str:
        if not self.apply_chat_template:
            if system or len(messages) != 1:
                raise ValueError("system prompts and multi-turn input require apply_chat_template=True")
            return messages[0]["content"]
        chat = ([{"role": "system", "content": system}] if system else []) + messages
        try:
            rendered = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                                          **self.chat_template_kwargs)
        except TypeError:
            self._warn_once("chat_template_kwargs",
                            f"{self.model}: template ignores {sorted(self.chat_template_kwargs)}")
            rendered = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if self.generation_prefix and "</think>" not in rendered[-64:]:
            rendered += self.generation_prefix
        return rendered

    def encode(self, texts: Sequence[str], **kwargs: Any):
        """An empty batch never reaches the tokenizer, which cannot describe one."""
        texts = list(texts)
        if not texts:
            return {"input_ids": [], "offset_mapping": []}
        return self.tokenizer(texts, add_special_tokens=not self.apply_chat_template, **kwargs)

    def score(self, prompts: Sequence[PromptLike], continuations: Sequence[str],
              prefixes: Optional[Sequence[str]] = None, system: Optional[str] = None) -> List[List[float]]:
        """Per-token log p(continuation | rendered prompt + assistant prefix); duplicate requests run once."""
        prefixes = prefixes if prefixes is not None else [""] * len(prompts)
        if not len(prompts) == len(continuations) == len(prefixes):
            raise ValueError("prompts, continuations and prefixes must have equal length")
        contexts = [self.render(*normalize_prompt(p, system)) + prefix for p, prefix in zip(prompts, prefixes)]
        requests = list(zip(contexts, continuations))
        scored: Dict[Tuple[str, str], List[float]] = {}
        for request in set(requests):
            hit = self.cache.get(self._score_key(*request)) if self.cache else None
            if hit is not None:
                scored[request] = hit["logprobs"]
        todo = sorted(set(requests) - scored.keys(), key=lambda r: len(r[0]) + len(r[1]), reverse=True)
        fresh = self._score_unique([c for c, _ in todo], [t for _, t in todo]) if todo else []
        for request, logprobs in zip(todo, fresh):
            scored[request] = logprobs
            if self.cache:
                self.cache.set(self._score_key(*request), {"logprobs": logprobs})
        return [scored[r] for r in requests]

    def _score_key(self, context: str, continuation: str) -> str:
        payload = {"kind": "score", "provider": self.provider, "model": self.model,
                   "context": context, "continuation": continuation}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def continuation_ids(self, contexts: Sequence[str], continuations: Sequence[str]) -> Tuple[List[List[int]], List[int]]:
        encoded = self.encode([c + t for c, t in zip(contexts, continuations)], return_offsets_mapping=True)
        lengths = [sum(end > len(context) for _, end in offsets)
                   for context, offsets in zip(contexts, encoded["offset_mapping"])]
        return encoded["input_ids"], lengths

    async def _run_batch(self, conversations: List[Tuple[Optional[str], Conversation]],
                         params: Dict[str, Any], n: int) -> List[List[Generation]]:
        params = self.effective_params(params)
        out: List[List[Optional[Generation]]] = [[None] * n for _ in conversations]
        pending = []
        for i, (system, messages) in enumerate(conversations):
            keys = [self.cache_key(system, messages, params, s) if self.cache else None for s in range(n)]
            missing = []
            for sample, key in enumerate(keys):
                hit = self.cache.get(key) if key else None
                if hit is None:
                    missing.append(sample)
                else:
                    out[i][sample] = Generation(**hit, cached=True)
            if missing:
                pending.append((i, missing, keys, self.render(system, messages)))
        if not pending:
            return out

        copies = max(len(item[1]) for item in pending)
        try:
            groups = await asyncio.to_thread(self._generate_texts, [item[3] for item in pending], params, copies)
        except Exception as exc:
            groups = [[self.failed(exc)] * copies for _ in pending]
        for (i, missing, keys, _), group in zip(pending, groups):
            for sample, generation in zip(missing, group):
                out[i][sample] = generation
                if keys[sample] and generation.ok:
                    self.cache.set(keys[sample], generation.to_dict())
        return out

    def _generate_texts(self, texts: List[str], params: Dict[str, Any], copies: int) -> List[List[Generation]]:
        raise NotImplementedError

    def _score_unique(self, contexts: List[str], continuations: List[str]) -> List[List[float]]:
        raise NotImplementedError
