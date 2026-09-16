"""One call syntax for every LLM: `LLM("provider/model")`, with API keys read from the environment."""

from __future__ import annotations

from typing import Any, Tuple

from .core import BaseLLM, Generation, LLMError
from .hf_models import HF_models
from .providers import OPENAI_COMPATIBLE

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

__all__ = ["LLM", "resolve_model", "BaseLLM", "Generation", "LLMError", "HF_models"]

_ALIASES = {"claude": "anthropic", "google": "gemini", "huggingface": "hf"}
_NATIVE = {"anthropic", "gemini", "hf", *_ALIASES}
_PREFIXES = (("openai", ("gpt-", "chatgpt", "o1", "o3", "o4")), ("anthropic", ("claude",)), ("gemini", ("gemini",)))


def resolve_model(model: str) -> Tuple[str, str]:
    head, _, rest = model.partition("/")
    if rest and (head.lower() in _NATIVE or head.lower() in OPENAI_COMPATIBLE):
        return _ALIASES.get(head.lower(), head.lower()), rest
    for provider, prefixes in _PREFIXES:
        if model.lower().startswith(prefixes):
            return provider, model
    if model in HF_models or "/" in model:
        return "hf", model
    raise ValueError(f"cannot infer provider for {model!r}; use 'provider/model'")


def LLM(model: str, **kwargs: Any) -> BaseLLM:
    provider, name = resolve_model(model)
    if provider == "anthropic":
        from .providers import AnthropicChat
        return AnthropicChat(name, **kwargs)
    if provider == "gemini":
        from .providers import GeminiChat
        return GeminiChat(name, **kwargs)
    if provider == "hf":
        from .hf import HFLocal
        return HFLocal(name, **kwargs)
    from .providers import OpenAIChat
    return OpenAIChat(name, provider=provider, **kwargs)
