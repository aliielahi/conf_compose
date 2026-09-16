"""Local HuggingFace causal LMs behind the same `generate()` / `prompt()` interface.

Env: `HF_TOKEN` for gated checkpoints (read by transformers automatically).
Batches prompts on the GPU and returns per-token logprobs when `logprobs=True`.
For scoring fixed completions (null-baselined class probabilities etc.) use `local_llm.LLM`.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from .core import BaseLLM, Conversation, Generation
from .hf_models import HF_models


class HFLocal(BaseLLM):
    provider = "hf"

    def __init__(self, model: str, *, batch_size: int = 16, dtype: Any = None, device: Optional[str] = None,
                 apply_chat_template: bool = True, **kwargs: Any):
        """
        Args:
            model: hub id or a short alias from `HF_models` (e.g. "l31-8bi").
            apply_chat_template: format prompts with the tokenizer's chat template. Set False to feed
                raw strings exactly as given (system prompts / multi-turn then require it to be True).
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs.setdefault("progress", True)
        super().__init__(HF_models.get(model, model), **kwargs)
        self.batch_size = batch_size
        self.apply_chat_template = apply_chat_template
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model,
            dtype=dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32),
            device_map=self.device if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if self.device != "cuda":
            self.hf_model.to(self.device)
        self.hf_model.eval()

        eos = self.hf_model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self._eos_ids = {i for i in eos + [self.tokenizer.eos_token_id] if i is not None}

    def unload(self) -> None:
        import gc
        import torch
        del self.hf_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # One GPU: no retries/concurrency — just cache lookup + batched generation off the event loop.
    async def _run_batch(self, convs: List[Tuple[Optional[str], Conversation]],
                         params: Dict[str, Any], n: int) -> List[List[Generation]]:
        params = self._effective_params(params)
        out: List[List[Optional[Generation]]] = [[None] * n for _ in convs]
        todo = []
        for i, (system, messages) in enumerate(convs):
            for s in range(n):
                key = self._cache_key(system, messages, params, s) if self.cache else None
                hit = self.cache.get(key) if key else None
                if hit is not None:
                    out[i][s] = Generation(**hit, cached=True)
                else:
                    todo.append((i, s, key, self._render(system, messages)))

        bar = self._bar(len(todo))
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start:start + self.batch_size]
            try:
                gens = await asyncio.to_thread(self._generate_batch, [c[3] for c in chunk], params)
            except Exception as exc:  # noqa: BLE001 — e.g. CUDA OOM; surfaced via on_error
                gens = [self._failed(exc)] * len(chunk)
            for (i, s, key, _), gen in zip(chunk, gens):
                out[i][s] = gen
                if key and gen.ok:
                    self.cache.set(key, gen.to_dict())
            if bar is not None:
                bar.update(len(chunk))
        if bar is not None:
            bar.close()
        return out  # type: ignore[return-value]

    def _render(self, system: Optional[str], messages: Conversation) -> str:
        if not self.apply_chat_template:
            if system or len(messages) != 1:
                raise ValueError("System prompts / multi-turn input need apply_chat_template=True.")
            return messages[0]["content"]
        chat = ([{"role": "system", "content": system}] if system else []) + messages
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def _generate_batch(self, texts: List[str], params: Dict[str, Any]) -> List[Generation]:
        import torch

        p = dict(params)
        want_logprobs = bool(p.pop("logprobs", False))
        temperature = p.pop("temperature", 0.0) or 0.0
        gen_args: Dict[str, Any] = {
            "max_new_tokens": p.pop("max_tokens", 256),
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": want_logprobs,
        }
        if temperature > 0:
            gen_args["temperature"] = temperature
            for k in ("top_p", "top_k"):
                if k in p:
                    gen_args[k] = p.pop(k)
        else:
            p.pop("top_p", None), p.pop("top_k", None)
        if "stop" in p:
            stop = p.pop("stop")
            gen_args["stop_strings"] = [stop] if isinstance(stop, str) else stop
            gen_args["tokenizer"] = self.tokenizer
        if "seed" in p:
            torch.manual_seed(p.pop("seed"))
        gen_args.update(p)

        enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                             add_special_tokens=not self.apply_chat_template).to(self.hf_model.device)
        with torch.no_grad():
            out = self.hf_model.generate(**enc, **gen_args)
        new_tokens = out.sequences[:, enc["input_ids"].shape[1]:]
        scores = (self.hf_model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
                  if want_logprobs else None)

        gens = []
        for j, ids in enumerate(new_tokens.tolist()):
            length = next((k for k, t in enumerate(ids) if t in self._eos_ids), None)
            finish = "stop" if length is not None else "length"
            ids = ids[:length] if length is not None else ids
            gens.append(Generation(
                text=self.tokenizer.decode(ids, skip_special_tokens=True),
                model=self.model,
                finish_reason=finish,
                logprobs=scores[j, :len(ids)].float().cpu().tolist() if scores is not None else None,
                tokens=self.tokenizer.convert_ids_to_tokens(ids) if want_logprobs else None,
                input_tokens=int(enc["attention_mask"][j].sum()),
                output_tokens=len(ids),
            ))
        del enc, out, scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return gens
