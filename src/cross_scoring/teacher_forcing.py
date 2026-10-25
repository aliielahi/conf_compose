"""Chat-formatted teacher forcing: token ids tagged by response span, and the scorer's log-probabilities of them."""

import numpy as np
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

MAX_TOKENS = 8192
# Prompts per generate call; prompt log-probs of a whole split at once exhaust the GB10's shared memory.
CHUNK = 100
# Prompt log-probs hold a full-vocabulary row per batched token (256k for Gemma), outside vLLM's memory budget.
BATCHED_TOKENS = 4096


def load(model):
    """vLLM engine, tokenizer and context limit of a scorer."""
    limit = min(MAX_TOKENS, getattr(AutoConfig.from_pretrained(model), "max_position_embeddings", MAX_TOKENS))
    llm = LLM(model, max_model_len=limit, gpu_memory_utilization=0.5,
              max_num_batched_tokens=BATCHED_TOKENS, trust_remote_code=True)
    return llm, AutoTokenizer.from_pretrained(model, trust_remote_code=True), limit


def encode(tokenizer, problem, response, starts):
    """Token ids of the chat and, per token, the index of the response span it starts in (-1 outside the response).

    starts are the character offsets where spans begin; the last span runs to the end of the response.
    """
    text = tokenizer.apply_chat_template([{"role": "user", "content": problem},
                                          {"role": "assistant", "content": response}], tokenize=False)
    offset = text.rindex(response)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    span = np.full(len(encoded.input_ids), -1)
    for i, (begin, _) in enumerate(encoded.offset_mapping):
        if 0 <= begin - offset < len(response):
            span[i] = np.searchsorted(starts, begin - offset, side="right") - 1
    return encoded.input_ids, span


def logprobs(llm, prompts, wanted):
    """Per prompt, (log-prob of the token, log-prob of the scorer's top token) at the positions in wanted."""
    out = []
    for start in range(0, len(prompts), CHUNK):
        outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompts[start:start + CHUNK]],
                               SamplingParams(max_tokens=1, prompt_logprobs=1), use_tqdm=False)
        for ids, positions, output in zip(prompts[start:start + CHUNK], wanted[start:start + CHUNK], outputs):
            candidates = [output.prompt_logprobs[t] for t in positions]
            out.append((np.array([c[ids[t]].logprob for c, t in zip(candidates, positions)], dtype=np.float32),
                        np.array([max(x.logprob for x in c.values()) for c in candidates], dtype=np.float32)))
    return out
