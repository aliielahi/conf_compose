"""API backends: OpenAI (+ any OpenAI-compatible server), Anthropic, Google Gemini.

Each uses the vendor's official async SDK with SDK-level retries disabled (retries are handled
uniformly in `core.BaseLLM`). API keys are read from the environment — see `OPENAI_COMPATIBLE`
and the class docstrings for variable names.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, Optional, Sequence, Tuple

from .core import BaseLLM, Conversation, Generation, retry_after_seconds

_RETRYABLE_STATUS = (408, 409, 425, 429)


def _retryable_status(code: Optional[int]) -> bool:
    return code is not None and (code in _RETRYABLE_STATUS or code >= 500)


def _env(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _transport_errors() -> Tuple[type, ...]:
    """Network/timeout exception classes of whichever HTTP stacks are installed."""
    errors: list = [asyncio.TimeoutError, ConnectionError]
    for module, names in (("httpx", ("TimeoutException", "TransportError")),
                          ("httpx2", ("TimeoutException", "TransportError")),
                          ("aiohttp", ("ClientConnectionError", "ServerTimeoutError"))):
        try:
            mod = __import__(module)
        except ImportError:
            continue
        errors += [getattr(mod, n) for n in names if hasattr(mod, n)]
    return tuple(errors)


def _as_list(stop: Any) -> Any:
    return [stop] if isinstance(stop, str) else stop


# ------------------------------------------------------------------------------ OpenAI
# provider -> (api key env var, default base url, key required)
OPENAI_COMPATIBLE: Dict[str, Tuple[str, Optional[str], bool]] = {
    "openai":     ("OPENAI_API_KEY", None, True),
    "deepseek":   ("DEEPSEEK_API_KEY", "https://api.deepseek.com", True),
    "together":   ("TOGETHER_API_KEY", "https://api.together.xyz/v1", True),
    "groq":       ("GROQ_API_KEY", "https://api.groq.com/openai/v1", True),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", True),
    "xai":        ("XAI_API_KEY", "https://api.x.ai/v1", True),
    "mistral":    ("MISTRAL_API_KEY", "https://api.mistral.ai/v1", True),
    "fireworks":  ("FIREWORKS_API_KEY", "https://api.fireworks.ai/inference/v1", True),
    "vllm":       ("VLLM_API_KEY", "http://localhost:8000/v1", False),
    "ollama":     ("OLLAMA_API_KEY", "http://localhost:11434/v1", False),
}


class OpenAIChat(BaseLLM):
    """Chat Completions API. Works for OpenAI and every server in `OPENAI_COMPATIBLE`.

    Env: `<PROVIDER>_API_KEY`; base URL overridable via `<PROVIDER>_BASE_URL` or `base_url=`.
    """

    def __init__(self, model: str, *, provider: str = "openai", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, **kwargs: Any):
        env_key, default_url, required = OPENAI_COMPATIBLE.get(provider, (f"{provider.upper()}_API_KEY", None, False))
        self.provider = provider
        self.api_key = api_key or _env(env_key) or (None if required else "not-needed")
        if self.api_key is None:
            raise EnvironmentError(f"Set {env_key} to use {provider} models.")
        self.base_url = base_url or _env(f"{provider.upper()}_BASE_URL") or default_url
        # OpenAI deprecated `max_tokens` (reasoning models reject it); most compatible servers only know `max_tokens`.
        self._max_tokens_field = "max_completion_tokens" if provider == "openai" else "max_tokens"
        super().__init__(model, **kwargs)

    def _make_client(self):
        import openai
        return openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url,
                                  max_retries=0, timeout=self.timeout)

    async def _agenerate_one(self, client, system: Optional[str], messages: Conversation,
                             params: Dict[str, Any]) -> Generation:
        p = dict(params)
        req: Dict[str, Any] = {"model": self.model,
                               "messages": ([{"role": "system", "content": system}] if system else []) + messages}
        if "max_tokens" in p:
            req[self._max_tokens_field] = p.pop("max_tokens")
        if "stop" in p:
            req["stop"] = _as_list(p.pop("stop"))
        if p.pop("logprobs", False):
            req["logprobs"] = True
        if "top_k" in p:  # not part of the OpenAI schema; compatible servers (vLLM, Together) accept it
            req.setdefault("extra_body", {})["top_k"] = p.pop("top_k")
        req.update(p)  # temperature, top_p, seed, and any provider-specific passthrough

        r = await client.chat.completions.create(**req)
        choice = r.choices[0]
        logprobs = tokens = None
        if getattr(choice, "logprobs", None) is not None and choice.logprobs.content:
            logprobs = [t.logprob for t in choice.logprobs.content]
            tokens = [t.token for t in choice.logprobs.content]
        usage = r.usage
        return Generation(
            text=choice.message.content or "",
            model=r.model or self.model,
            finish_reason=choice.finish_reason,
            logprobs=logprobs,
            tokens=tokens,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            raw=r,
        )

    def _classify_error(self, exc: BaseException):
        import openai
        if isinstance(exc, (asyncio.TimeoutError, openai.APITimeoutError, openai.APIConnectionError)):
            return True, None
        if isinstance(exc, openai.APIStatusError):
            if getattr(exc, "code", None) == "insufficient_quota":  # 429 that will never succeed
                return False, None
            return _retryable_status(exc.status_code), retry_after_seconds(exc.response.headers)
        return False, None


# ------------------------------------------------------------------------------ Anthropic
class AnthropicChat(BaseLLM):
    """Claude via the Messages API.

    Env: `ANTHROPIC_API_KEY` (the SDK also accepts `ANTHROPIC_AUTH_TOKEN` or an `ant auth login` profile).
    Per-token logprobs are not available from this API.
    """

    provider = "anthropic"
    DEFAULT_MAX_TOKENS = 16000  # required by the API; generous so thinking + answer aren't truncated

    # Models that reject sampling params (temperature/top_p/top_k return a 400).
    _NO_SAMPLING = re.compile(r"claude-(opus-(4-[78]|5)|sonnet-5|fable|mythos)")

    def __init__(self, model: str, *, api_key: Optional[str] = None, **kwargs: Any):
        self.api_key = api_key
        super().__init__(model, **kwargs)

    def _unsupported_params(self) -> Sequence[str]:
        dropped = ["logprobs", "seed"]
        if self._NO_SAMPLING.search(self.model):
            dropped += ["temperature", "top_p", "top_k"]
        return dropped

    def _make_client(self):
        import anthropic
        return anthropic.AsyncAnthropic(api_key=self.api_key, max_retries=0, timeout=self.timeout)

    async def _agenerate_one(self, client, system: Optional[str], messages: Conversation,
                             params: Dict[str, Any]) -> Generation:
        p = dict(params)
        req: Dict[str, Any] = {"model": self.model, "messages": messages,
                               "max_tokens": p.pop("max_tokens", self.DEFAULT_MAX_TOKENS)}
        if system:
            req["system"] = system
        if "stop" in p:
            req["stop_sequences"] = _as_list(p.pop("stop"))
        # SDK 1.x removed these kwargs; the API still honours them on models that accept sampling params.
        sampling = {k: p.pop(k) for k in ("temperature", "top_p", "top_k") if k in p}
        if sampling:
            req["extra_body"] = {**p.pop("extra_body", {}), **sampling}
        req.update(p)  # thinking, output_config, metadata, ...

        r = await client.messages.create(**req)
        text = "".join(block.text for block in r.content if block.type == "text")
        return Generation(
            text=text,
            model=r.model,
            finish_reason=r.stop_reason,  # "end_turn" | "max_tokens" | "stop_sequence" | "refusal" | ...
            input_tokens=r.usage.input_tokens,
            output_tokens=r.usage.output_tokens,
            raw=r,
        )

    def _classify_error(self, exc: BaseException):
        import anthropic
        if isinstance(exc, (asyncio.TimeoutError, anthropic.APITimeoutError, anthropic.APIConnectionError)):
            return True, None
        if isinstance(exc, anthropic.APIStatusError):  # 529 overloaded is covered by >= 500
            return _retryable_status(exc.status_code), retry_after_seconds(exc.response.headers)
        return False, None


# ------------------------------------------------------------------------------ Gemini
class GeminiChat(BaseLLM):
    """Gemini via the `google-genai` SDK.

    Env: `GEMINI_API_KEY` or `GOOGLE_API_KEY`. Provider-specific fields (e.g. `tools`,
    `thinking_config`, `safety_settings`) are passed into `GenerateContentConfig`.
    """

    provider = "gemini"

    def __init__(self, model: str, *, api_key: Optional[str] = None, **kwargs: Any):
        self.api_key = api_key or _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
        if self.api_key is None:
            raise EnvironmentError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use Gemini models.")
        super().__init__(model, **kwargs)

    def _make_client(self):
        from google import genai
        from google.genai import types
        return genai.Client(api_key=self.api_key,
                            http_options=types.HttpOptions(timeout=int(self.timeout * 1000)))

    async def _agenerate_one(self, client, system: Optional[str], messages: Conversation,
                             params: Dict[str, Any]) -> Generation:
        from google.genai import types
        p = dict(params)
        contents = [types.Content(role="model" if m["role"] == "assistant" else "user",
                                  parts=[types.Part(text=m["content"])]) for m in messages]
        config: Dict[str, Any] = {"system_instruction": system}
        if "max_tokens" in p:
            config["max_output_tokens"] = p.pop("max_tokens")
        if "stop" in p:
            config["stop_sequences"] = _as_list(p.pop("stop"))
        if p.pop("logprobs", False):
            config["response_logprobs"] = True
        config.update(p)  # temperature, top_p, top_k, seed, tools, thinking_config, ...
        config = {k: v for k, v in config.items() if v is not None}

        r = await client.aio.models.generate_content(
            model=self.model, contents=contents, config=types.GenerateContentConfig(**config))

        cand = r.candidates[0] if r.candidates else None
        text, finish, logprobs, tokens = "", None, None, None
        if cand is not None:
            parts = (cand.content.parts if cand.content and cand.content.parts else [])
            text = "".join(part.text for part in parts if part.text and not getattr(part, "thought", False))
            finish = cand.finish_reason.name if cand.finish_reason else None
            lp = getattr(cand, "logprobs_result", None)
            if lp is not None and lp.chosen_candidates:
                logprobs = [c.log_probability for c in lp.chosen_candidates]
                tokens = [c.token for c in lp.chosen_candidates]
        elif r.prompt_feedback is not None and r.prompt_feedback.block_reason:
            finish = f"blocked:{r.prompt_feedback.block_reason.name}"
        usage = r.usage_metadata
        return Generation(
            text=text,
            model=self.model,
            finish_reason=finish,
            logprobs=logprobs,
            tokens=tokens,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            raw=r,
        )

    def _classify_error(self, exc: BaseException):
        from google.genai import errors
        if isinstance(exc, _transport_errors()):
            return True, None
        if isinstance(exc, errors.APIError):
            headers = getattr(getattr(exc, "response", None), "headers", None)
            return _retryable_status(exc.code), retry_after_seconds(headers)
        return False, None
