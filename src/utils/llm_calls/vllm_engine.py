"""vLLM in-process backend: continuous batching, prefix caching, n-sample generation, prompt-logprob scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .core import Generation
from .local import LocalLLM


class VLLMLocal(LocalLLM):
    provider = "vllm"

    def __init__(self, model: str, *, gpu_memory_utilization: float = 0.9, max_model_len: Optional[int] = 8192,
                 tensor_parallel_size: int = 1, dtype: str = "auto", seed: int = 0,
                 engine_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any):
        from vllm import LLM as Engine

        super().__init__(model, **kwargs)
        self.engine = Engine(model=self.model, dtype=dtype, seed=seed, max_model_len=max_model_len,
                             gpu_memory_utilization=gpu_memory_utilization,
                             tensor_parallel_size=tensor_parallel_size, enable_prefix_caching=True,
                             **(engine_kwargs or {}))
        self.tokenizer = self.engine.get_tokenizer()

    def _generate_texts(self, texts: List[str], params: Dict[str, Any], copies: int) -> List[List[Generation]]:
        from vllm import SamplingParams

        params = dict(params)
        with_logprobs = bool(params.pop("logprobs", False))
        temperature = params.pop("temperature", 0.0) or 0.0
        sampling = temperature > 0
        options: Dict[str, Any] = {"n": copies if sampling else 1, "temperature": temperature,
                                   "max_tokens": params.pop("max_tokens", 256)}
        if with_logprobs:
            options["logprobs"] = 1
        if "stop" in params:
            stop = params.pop("stop")
            options["stop"] = [stop] if isinstance(stop, str) else stop
        options.update(params)

        prompts = [{"prompt_token_ids": ids} for ids in self.encode(texts)["input_ids"]]
        outputs = self.engine.generate(prompts, SamplingParams(**options), use_tqdm=self.progress)

        groups = []
        for output in outputs:
            group = [_to_generation(self.model, completion, len(output.prompt_token_ids), with_logprobs)
                     for completion in output.outputs]
            groups.append(group if sampling else group * copies)
        return groups

    def _score_unique(self, contexts: List[str], continuations: List[str]) -> List[List[float]]:
        from vllm import SamplingParams

        token_ids, lengths = self.continuation_ids(contexts, continuations)
        prompts = [{"prompt_token_ids": ids} for ids in token_ids]
        outputs = self.engine.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=1),
                                       use_tqdm=self.progress)
        scores = []
        for output, ids, length in zip(outputs, token_ids, lengths):
            positions = range(len(ids) - length, len(ids))
            scores.append([output.prompt_logprobs[p][ids[p]].logprob for p in positions if p > 0])
        return scores


def _to_generation(model: str, completion, input_tokens: int, with_logprobs: bool) -> Generation:
    logprobs = tokens = None
    if with_logprobs and completion.logprobs:
        chosen = [step[token] for step, token in zip(completion.logprobs, completion.token_ids)]
        logprobs = [c.logprob for c in chosen]
        tokens = [c.decoded_token for c in chosen]
    return Generation(
        text=completion.text,
        model=model,
        finish_reason=completion.finish_reason,
        logprobs=logprobs,
        tokens=tokens,
        input_tokens=input_tokens,
        output_tokens=len(completion.token_ids),
    )
