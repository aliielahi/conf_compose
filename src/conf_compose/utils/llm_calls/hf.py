"""HuggingFace transformers backend: length-sorted static batching for generation and scoring."""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import Generation
from .local import LocalLLM


class HFLocal(LocalLLM):
    provider = "hf"

    def __init__(self, model: str, *, batch_size: int = 16, dtype: Any = None, device: Optional[str] = None,
                 **kwargs: Any):
        super().__init__(model, **kwargs)
        self.batch_size = batch_size
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

    def _score_unique(self, contexts: List[str], continuations: List[str]) -> List[List[float]]:
        scores: List[List[float]] = []
        for start in range(0, len(contexts), self.batch_size):
            batch = slice(start, start + self.batch_size)
            scores += self._score_batch(contexts[batch], continuations[batch])
        return scores

    def _score_batch(self, contexts: List[str], continuations: List[str]) -> List[List[float]]:
        encoded = self.encode([c + t for c, t in zip(contexts, continuations)], return_tensors="pt",
                              padding=True, return_offsets_mapping=True)
        token_ends, mask = encoded.pop("offset_mapping")[:, :, 1], encoded["attention_mask"].bool()
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

    def _generate_texts(self, texts: List[str], params: Dict[str, Any], copies: int) -> List[List[Generation]]:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        sampling = (params.get("temperature") or 0) > 0
        chunk_size = max(1, self.batch_size // copies) if sampling else self.batch_size
        groups: List[Optional[List[Generation]]] = [None] * len(texts)
        bar = self._progress_bar(len(texts))
        for start in range(0, len(order), chunk_size):
            chunk = order[start:start + chunk_size]
            try:
                results = self._generate_batch([texts[i] for i in chunk], params, copies)
            except Exception as exc:
                results = [[self.failed(exc)] * copies for _ in chunk]
            for i, group in zip(chunk, results):
                groups[i] = group
            if bar is not None:
                bar.update(len(chunk))
        if bar is not None:
            bar.close()
        return groups

    def _generate_batch(self, texts: List[str], params: Dict[str, Any], copies: int) -> List[List[Generation]]:
        params = dict(params)
        with_logprobs = bool(params.pop("logprobs", False))
        top_logprobs = params.pop("top_logprobs", 0) or 0
        temperature = params.pop("temperature", 0.0) or 0.0
        top_p, top_k = params.pop("top_p", None), params.pop("top_k", None)
        sampling = temperature > 0
        per_prompt = copies if sampling else 1
        options: Dict[str, Any] = {
            "max_new_tokens": params.pop("max_tokens", 256),
            "do_sample": sampling,
            "num_return_sequences": per_prompt,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": with_logprobs or bool(top_logprobs),
        }
        if sampling:
            options.update({k: v for k, v in (("temperature", temperature), ("top_p", top_p), ("top_k", top_k))
                            if v is not None})
        if "stop" in params:
            stop = params.pop("stop")
            options.update(stop_strings=[stop] if isinstance(stop, str) else stop, tokenizer=self.tokenizer)
        if "seed" in params:
            torch.manual_seed(params.pop("seed"))
        options.update(params)

        encoded = self.encode(texts, return_tensors="pt", padding=True).to(self.hf_model.device)
        with torch.no_grad():
            output = self.hf_model.generate(**encoded, **options)
        new_tokens = output.sequences[:, encoded["input_ids"].shape[1]:].tolist()
        transition = (self.hf_model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
                      if with_logprobs else None)
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()

        generations = []
        for row, ids in enumerate(new_tokens):
            stop_at = next((k for k, token in enumerate(ids) if token in self.eos_ids), None)
            ids = ids if stop_at is None else ids[:stop_at]
            generations.append(Generation(
                text=self.tokenizer.decode(ids, skip_special_tokens=True),
                model=self.model,
                finish_reason="length" if stop_at is None else "stop",
                logprobs=transition[row, :len(ids)].float().tolist() if with_logprobs else None,
                tokens=self.tokenizer.convert_ids_to_tokens(ids) if with_logprobs else None,
                top_logprobs=self._top_logprobs(output.scores, row, len(ids), top_logprobs) if top_logprobs else None,
                input_tokens=input_lengths[row // per_prompt],
                output_tokens=len(ids),
            ))
        del encoded, output, transition
        _free_cuda()
        if sampling:
            return [generations[j:j + copies] for j in range(0, len(generations), copies)]
        return [[generation] * copies for generation in generations]

    def _top_logprobs(self, scores, row: int, length: int, k: int) -> List[Dict[str, float]]:
        steps = []
        for step in scores[:length]:
            values, indices = torch.log_softmax(step[row].float(), dim=-1).topk(k)
            top: Dict[str, float] = {}
            for token, logprob in zip(self.tokenizer.batch_decode(indices[:, None]), values.tolist()):
                top[token] = max(logprob, top.get(token, float("-inf")))
            steps.append(top)
        return steps


def _free_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
