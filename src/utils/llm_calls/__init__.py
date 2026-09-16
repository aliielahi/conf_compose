"""One interface for every LLM: OpenAI, Anthropic, Gemini, OpenAI-compatible servers, local HF.

    from llm_calls import LLM

    llm = LLM("openai/gpt-4o-mini")            # or "claude-opus-5", "gemini/gemini-2.5-flash",
                                               # "together/meta-llama/...", "vllm/<served-name>",
                                               # "hf/l31-8bi" (local transformers)
    llm("What is 2+2?")                        # -> "4"
    llm(["q1", "q2"], temperature=0.3, n=3)    # -> [[s1, s2, s3], [s1, s2, s3]]
    g = llm.generate("q", logprobs=True)       # -> Generation(text, logprobs, finish_reason, usage, ...)
    await llm.agenerate([...])                 # async variant

API keys come from the environment (a `.env` file is loaded if python-dotenv is installed):
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY | GOOGLE_API_KEY, DEEPSEEK_API_KEY,
    TOGETHER_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, XAI_API_KEY, MISTRAL_API_KEY,
    FIREWORKS_API_KEY, HF_TOKEN; local servers via VLLM_BASE_URL / OLLAMA_BASE_URL.

Backends import lazily, so each only needs its own SDK installed.
Legacy: `LanguageModelClient` (api_clients.py) and `LocalLLM` (local_llm.LLM, likelihood scoring).
"""

from __future__ import annotations

from typing import Any, Tuple

from .core import BaseLLM, Generation, LLMError
from .hf_models import HF_models

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

__all__ = ["LLM", "resolve_model", "BaseLLM", "Generation", "LLMError", "HF_models",
           "LanguageModelClient", "LocalLLM"]

_NATIVE = {"openai", "anthropic", "claude", "gemini", "google", "hf", "huggingface"}


def resolve_model(model: str) -> Tuple[str, str]:
    """"provider/model" -> (provider, model). The provider prefix may be omitted for obvious names."""
    from .providers import OPENAI_COMPATIBLE

    if "/" in model:
        head, rest = model.split("/", 1)
        if head.lower() in _NATIVE or head.lower() in OPENAI_COMPATIBLE:
            provider = {"claude": "anthropic", "google": "gemini", "huggingface": "hf"}.get(head.lower(), head.lower())
            return provider, rest
    lowered = model.lower()
    if lowered.startswith(("gpt-", "chatgpt", "o1", "o3", "o4")):
        return "openai", model
    if lowered.startswith("claude"):
        return "anthropic", model
    if lowered.startswith("gemini"):
        return "gemini", model
    if model in HF_models or "/" in model:
        return "hf", model
    raise ValueError(f"Can't infer the provider for {model!r}; use 'provider/model' "
                     f"(providers: {sorted(_NATIVE | set(OPENAI_COMPATIBLE))}).")


def LLM(model: str, **kwargs: Any) -> BaseLLM:
    """Build a client for `model` ("provider/model"). kwargs: see `BaseLLM.__init__` and the backend class."""
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


def __getattr__(name):
    if name == "LanguageModelClient":
        from .api_clients import LanguageModelClient
        return LanguageModelClient
    if name == "LocalLLM":
        from .local_llm import LLM as LocalLLM
        return LocalLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
