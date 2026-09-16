"""Local HuggingFace causal LMs: batched generation and teacher-forced continuation scoring."""

from __future__ import annotations

import asyncio
import gc
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import BaseLLM, Conversation, Generation, PromptLike, normalize_prompt
from .hf_models import HF_models


class HFLocal(BaseLLM):
    provider = "hf"

    def __init__(self, model: str, *, batch_size: int = 16, dtype: Any = None, device: Optional[str] = None,
                 apply_chat_template: bool = True, **kwargs: Any):
        super().__init__(HF_models.get(model, model), **kwargs)
        self.batch_size = batch_size
        self.apply_chat_template = apply_chat_template
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        on_gpu = self.device == "cuda"
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model,
            dtype=dtype or (torch.bfloat16 if on_gpu else torch.float32),
            device_map="cuda" if on_gpu else None,
            low_cpu_mem_usage=True,
        )
        if not on_gpu:
            self.hf_model.to(self.device)
        self.hf_model.eval()

        eos = self.hf_model.generation_config.eos_token_id
        self.eos_ids = {i for i in [*(eos if isinstance(eos, list) else [eos]), self.tokenizer.eos_token_id]
                        if i is not None}

    def unload(self) -> None:
        del self.hf_model
        gc.collect()
        _free_cuda()

    def render(self, system: Optional[str], messages: Conversation) -> str:
        if not self.apply_chat_template:
            if system or len(messages) != 1:
                raise ValueError("system prompts and multi-turn input require apply_chat_template=True")
            return messages[0]["content"]
        chat = ([{"role": "system", "content": system}] if system else []) + messages
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def score(self, prompts: Sequence[PromptLike], continuations: Sequence[str],
              prefixes: Optional[Sequence[str]] = None, system: Optional[str] = None,
              batch_size: Optional[int] = None) -> List[List[float]]:
        """Per-token log p(continuation | rendered prompt + assistant prefix), located by character offsets."""
        prefixes = prefixes if prefixes is not None else [""] * len(prompts)
        if not len(prompts) == len(continuations) == len(prefixes):
            raise ValueError("prompts, continuations and prefixes must have equal length")
        contexts = [self.render(*normalize_prompt(p, system)) + prefix for p, prefix in zip(prompts, prefixes)]
        batch_size = batch_size or self.batch_size
        scores: List[List[float]] = []
        for start in range(0, len(contexts), batch_size):
            batch = slice(start, start + batch_size)
            scores += self._score_batch(contexts[batch], continuations[batch])
        return scores

    def _score_batch(self, contexts: Sequence[str], continuations: Sequence[str]) -> List[List[float]]:
        texts = [context + continuation for context, continuation in zip(contexts, continuations)]
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True, return_offsets_mapping=True,
                                 add_special_tokens=not self.apply_chat_template)
        token_ends = encoded.pop("offset_mapping")[:, :, 1]
        mask = encoded["attention_mask"].bool()
        lengths = [int(((token_ends[i] > len(context)) & mask[i]).sum()) for i, context in enumerate(contexts)]
        encoded = encoded.to(self.hf_model.device)
        with torch.no_grad():
            logits = self.hf_model(**encoded, logits_to_keep=max(lengths) + 1).logits

        scores = []
        for i, length in enumerate(lengths):
            if length == 0:
                scores.append([])
                continue
            logprobs = torch.log_softmax(logits[i, -length - 1:-1].float(), dim=-1)
            targets = encoded["input_ids"][i, -length:, None]
            scores.append(logprobs.gather(-1, targets).squeeze(-1).tolist())
        del encoded, logits
        _free_cuda()
        return scores

    async def _run_batch(self, conversations: List[Tuple[Optional[str], Conversation]],
                         params: Dict[str, Any], n: int) -> List[List[Generation]]:
        params = self.effective_params(params)
        out: List[List[Optional[Generation]]] = [[None] * n for _ in conversations]
        pending = []
        for i, (system, messages) in enumerate(conversations):
            for sample in range(n):
                key = self.cache_key(system, messages, params, sample) if self.cache else None
                hit = self.cache.get(key) if key else None
                if hit is not None:
                    out[i][sample] = Generation(**hit, cached=True)
                else:
                    pending.append((i, sample, key, self.render(system, messages)))

        bar = self._progress_bar(len(pending))
        for start in range(0, len(pending), self.batch_size):
            chunk = pending[start:start + self.batch_size]
            try:
                generations = await asyncio.to_thread(self._generate_batch, [c[3] for c in chunk], params)
            except Exception as exc:
                generations = [self.failed(exc)] * len(chunk)
            for (i, sample, key, _), generation in zip(chunk, generations):
                out[i][sample] = generation
                if key and generation.ok:
                    self.cache.set(key, generation.to_dict())
            if bar is not None:
                bar.update(len(chunk))
        if bar is not None:
            bar.close()
        return out

    def _generate_batch(self, texts: List[str], params: Dict[str, Any]) -> List[Generation]:
        params = dict(params)
        with_logprobs = bool(params.pop("logprobs", False))
        temperature = params.pop("temperature", 0.0) or 0.0
        top_p, top_k = params.pop("top_p", None), params.pop("top_k", None)
        options: Dict[str, Any] = {
            "max_new_tokens": params.pop("max_tokens", 256),
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": with_logprobs,
        }
        if temperature > 0:
            options.update({k: v for k, v in (("temperature", temperature), ("top_p", top_p), ("top_k", top_k))
                            if v is not None})
        if "stop" in params:
            stop = params.pop("stop")
            options.update(stop_strings=[stop] if isinstance(stop, str) else stop, tokenizer=self.tokenizer)
        if "seed" in params:
            torch.manual_seed(params.pop("seed"))
        options.update(params)

        encoded = self.tokenizer(texts, return_tensors="pt", padding=True,
                                 add_special_tokens=not self.apply_chat_template).to(self.hf_model.device)
        with torch.no_grad():
            output = self.hf_model.generate(**encoded, **options)
        new_tokens = output.sequences[:, encoded["input_ids"].shape[1]:].tolist()
        transition = (self.hf_model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
                      if with_logprobs else None)

        generations = []
        for i, ids in enumerate(new_tokens):
            stop_at = next((k for k, token in enumerate(ids) if token in self.eos_ids), None)
            ids = ids if stop_at is None else ids[:stop_at]
            generations.append(Generation(
                text=self.tokenizer.decode(ids, skip_special_tokens=True),
                model=self.model,
                finish_reason="length" if stop_at is None else "stop",
                logprobs=transition[i, :len(ids)].float().tolist() if with_logprobs else None,
                tokens=self.tokenizer.convert_ids_to_tokens(ids) if with_logprobs else None,
                input_tokens=int(encoded["attention_mask"][i].sum()),
                output_tokens=len(ids),
            ))
        del encoded, output, transition
        _free_cuda()
        return generations


def _free_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
