"""API backends: OpenAI and OpenAI-compatible servers, Anthropic, Google Gemini."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, Optional, Sequence, Tuple

from .core import BaseLLM, Conversation, Generation, retry_after_seconds

OPENAI_COMPATIBLE: Dict[str, Tuple[str, Optional[str], bool]] = {
    "openai": ("OPENAI_API_KEY", None, True),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com", True),
    "together": ("TOGETHER_API_KEY", "https://api.together.xyz/v1", True),
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1", True),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", True),
    "xai": ("XAI_API_KEY", "https://api.x.ai/v1", True),
    "mistral": ("MISTRAL_API_KEY", "https://api.mistral.ai/v1", True),
    "fireworks": ("FIREWORKS_API_KEY", "https://api.fireworks.ai/inference/v1", True),
    "vllm": ("VLLM_API_KEY", "http://localhost:8000/v1", False),
    "ollama": ("OLLAMA_API_KEY", "http://localhost:11434/v1", False),
}


def _env(*names: str) -> Optional[str]:
    return next((os.environ[name] for name in names if os.environ.get(name)), None)


def _as_list(value: Any) -> Any:
    return [value] if isinstance(value, str) else value


def _retryable_status(code: Optional[int]) -> bool:
    return code is not None and (code in (408, 409, 425, 429) or code >= 500)


def _transport_errors() -> Tuple[type, ...]:
    errors = [asyncio.TimeoutError, ConnectionError]
    for module, names in (("httpx", ("TimeoutException", "TransportError")),
                          ("httpx2", ("TimeoutException", "TransportError")),
                          ("aiohttp", ("ClientConnectionError", "ServerTimeoutError"))):
        try:
            imported = __import__(module)
        except ImportError:
            continue
        errors += [getattr(imported, name) for name in names if hasattr(imported, name)]
    return tuple(errors)


class OpenAIChat(BaseLLM):
    def __init__(self, model: str, *, provider: str = "openai", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, **kwargs: Any):
        env_key, default_url, key_required = OPENAI_COMPATIBLE.get(
            provider, (f"{provider.upper()}_API_KEY", None, False))
        self.provider = provider
        self.api_key = api_key or _env(env_key) or (None if key_required else "not-needed")
        if self.api_key is None:
            raise EnvironmentError(f"set {env_key} to use {provider} models")
        self.base_url = base_url or _env(f"{provider.upper()}_BASE_URL") or default_url
        self.max_tokens_field = "max_completion_tokens" if provider == "openai" else "max_tokens"
        super().__init__(model, **kwargs)

    def _make_client(self):
        import openai
        return openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0, timeout=self.timeout)

    async def _generate_one(self, client, system: Optional[str], messages: Conversation,
                            params: Dict[str, Any]) -> Generation:
        params = dict(params)
        system_messages = [{"role": "system", "content": system}] if system else []
        request: Dict[str, Any] = {"model": self.model, "messages": system_messages + messages}
        if "max_tokens" in params:
            request[self.max_tokens_field] = params.pop("max_tokens")
        if "stop" in params:
            request["stop"] = _as_list(params.pop("stop"))
        if params.pop("logprobs", False):
            request["logprobs"] = True
        if "top_k" in params:
            request.setdefault("extra_body", {})["top_k"] = params.pop("top_k")
        request.update(params)

        response = await client.chat.completions.create(**request)
        choice = response.choices[0]
        token_logprobs = choice.logprobs.content if choice.logprobs and choice.logprobs.content else None
        return Generation(
            text=choice.message.content or "",
            model=response.model or self.model,
            finish_reason=choice.finish_reason,
            logprobs=[t.logprob for t in token_logprobs] if token_logprobs else None,
            tokens=[t.token for t in token_logprobs] if token_logprobs else None,
            input_tokens=getattr(response.usage, "prompt_tokens", None),
            output_tokens=getattr(response.usage, "completion_tokens", None),
            raw=response,
        )

    def _classify_error(self, exc: BaseException):
        import openai
        if isinstance(exc, (asyncio.TimeoutError, openai.APITimeoutError, openai.APIConnectionError)):
            return True, None
        if isinstance(exc, openai.APIStatusError) and getattr(exc, "code", None) != "insufficient_quota":
            return _retryable_status(exc.status_code), retry_after_seconds(exc.response.headers)
        return False, None


class AnthropicChat(BaseLLM):
    provider = "anthropic"
    default_max_tokens = 16000
    no_sampling_models = re.compile(r"claude-(opus-(4-[78]|5)|sonnet-5|fable|mythos)")

    def __init__(self, model: str, *, api_key: Optional[str] = None, **kwargs: Any):
        self.api_key = api_key
        super().__init__(model, **kwargs)

    def _unsupported_params(self) -> Sequence[str]:
        sampling = ["temperature", "top_p", "top_k"] if self.no_sampling_models.search(self.model) else []
        return ["logprobs", "seed", *sampling]

    def _make_client(self):
        import anthropic
        return anthropic.AsyncAnthropic(api_key=self.api_key, max_retries=0, timeout=self.timeout)

    async def _generate_one(self, client, system: Optional[str], messages: Conversation,
                            params: Dict[str, Any]) -> Generation:
        params = dict(params)
        request: Dict[str, Any] = {"model": self.model, "messages": messages,
                                   "max_tokens": params.pop("max_tokens", self.default_max_tokens)}
        if system:
            request["system"] = system
        if "stop" in params:
            request["stop_sequences"] = _as_list(params.pop("stop"))
        sampling = {k: params.pop(k) for k in ("temperature", "top_p", "top_k") if k in params}
        if sampling:  # removed from SDK 1.x kwargs but still accepted by the API
            request["extra_body"] = {**params.pop("extra_body", {}), **sampling}
        request.update(params)

        response = await client.messages.create(**request)
        return Generation(
            text="".join(block.text for block in response.content if block.type == "text"),
            model=response.model,
            finish_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw=response,
        )

    def _classify_error(self, exc: BaseException):
        import anthropic
        if isinstance(exc, (asyncio.TimeoutError, anthropic.APITimeoutError, anthropic.APIConnectionError)):
            return True, None
        if isinstance(exc, anthropic.APIStatusError):
            return _retryable_status(exc.status_code), retry_after_seconds(exc.response.headers)
        return False, None


class GeminiChat(BaseLLM):
    provider = "gemini"

    def __init__(self, model: str, *, api_key: Optional[str] = None, **kwargs: Any):
        self.api_key = api_key or _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
        if self.api_key is None:
            raise EnvironmentError("set GEMINI_API_KEY or GOOGLE_API_KEY to use Gemini models")
        super().__init__(model, **kwargs)

    def _make_client(self):
        from google import genai
        from google.genai import types
        return genai.Client(api_key=self.api_key, http_options=types.HttpOptions(timeout=int(self.timeout * 1000)))

    async def _generate_one(self, client, system: Optional[str], messages: Conversation,
                            params: Dict[str, Any]) -> Generation:
        from google.genai import types

        params = dict(params)
        contents = [types.Content(role="model" if m["role"] == "assistant" else "user",
                                  parts=[types.Part(text=m["content"])]) for m in messages]
        config: Dict[str, Any] = {"system_instruction": system}
        if "max_tokens" in params:
            config["max_output_tokens"] = params.pop("max_tokens")
        if "stop" in params:
            config["stop_sequences"] = _as_list(params.pop("stop"))
        if params.pop("logprobs", False):
            config["response_logprobs"] = True
        config.update(params)
        config = {k: v for k, v in config.items() if v is not None}

        response = await client.aio.models.generate_content(
            model=self.model, contents=contents, config=types.GenerateContentConfig(**config))

        candidate = response.candidates[0] if response.candidates else None
        generation = Generation(text="", model=self.model, raw=response,
                                input_tokens=getattr(response.usage_metadata, "prompt_token_count", None),
                                output_tokens=getattr(response.usage_metadata, "candidates_token_count", None))
        if candidate is None:
            feedback = response.prompt_feedback
            if feedback is not None and feedback.block_reason:
                generation.finish_reason = f"blocked:{feedback.block_reason.name}"
            return generation
        parts = candidate.content.parts if candidate.content and candidate.content.parts else []
        generation.text = "".join(p.text for p in parts if p.text and not getattr(p, "thought", False))
        generation.finish_reason = candidate.finish_reason.name if candidate.finish_reason else None
        chosen = candidate.logprobs_result.chosen_candidates if candidate.logprobs_result else None
        if chosen:
            generation.logprobs = [c.log_probability for c in chosen]
            generation.tokens = [c.token for c in chosen]
        return generation

    def _classify_error(self, exc: BaseException):
        from google.genai import errors
        if isinstance(exc, _transport_errors()):
            return True, None
        if isinstance(exc, errors.APIError):
            headers = getattr(getattr(exc, "response", None), "headers", None)
            return _retryable_status(exc.code), retry_after_seconds(headers)
        return False, None
