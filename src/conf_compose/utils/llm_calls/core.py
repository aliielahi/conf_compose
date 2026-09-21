"""Provider-agnostic LLM client: call syntax, batching, retries, rate limits, caching, sync bridge."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import sqlite3
import threading
import time
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

Message = Dict[str, str]
Conversation = List[Message]
PromptLike = Union[str, Conversation]

_UNSUPPORTED = re.compile(
    r"unsupported|not supported|does not support|doesn't support|not allowed|not permitted|"
    r"deprecated|only the default|is not available|unknown parameter|unrecognized|extra inputs"
)
_PARAM_ALIASES = {
    "temperature": ("temperature",),
    "top_p": ("top_p", "top-p", "topp"),
    "top_k": ("top_k", "top-k", "topk"),
    "logprobs": ("logprob",),
    "top_logprobs": ("top_logprob", "logprob"),
    "seed": ("seed",),
    "stop": ("stop",),
}


class LLMError(RuntimeError):
    pass


@dataclass
class Generation:
    text: str
    model: str
    finish_reason: Optional[str] = None
    logprobs: Optional[List[float]] = None
    tokens: Optional[List[str]] = None
    top_logprobs: Optional[List[Dict[str, float]]] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None
    cached: bool = False
    raw: Any = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def mean_logprob(self) -> Optional[float]:
        return sum(self.logprobs) / len(self.logprobs) if self.logprobs else None

    def to_dict(self) -> Dict[str, Any]:
        keys = ("text", "model", "finish_reason", "logprobs", "tokens", "top_logprobs", "input_tokens", "output_tokens",
                "error")
        return {k: getattr(self, k) for k in keys}

    def __str__(self) -> str:
        return self.text


class _BackgroundLoop:
    _lock = threading.Lock()
    _loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def run(cls, coro):
        with cls._lock:
            if cls._loop is None:
                cls._loop = asyncio.new_event_loop()
                threading.Thread(target=cls._loop.run_forever, name="llm_calls", daemon=True).start()
        future = asyncio.run_coroutine_threadsafe(coro, cls._loop)
        try:
            return future.result()
        except KeyboardInterrupt:
            future.cancel()
            raise


class DiskCache:
    def __init__(self, path: Union[str, Path]):
        path = Path(path)
        if path.suffix != ".sqlite":
            path.mkdir(parents=True, exist_ok=True)
            path = path / "llm_cache.sqlite"
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)")
        self._db.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
            self.hits += row is not None
            self.misses += row is None
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value)))
            self._db.commit()


class _LoopState:
    def __init__(self, loop: asyncio.AbstractEventLoop, client: Any, concurrency: int):
        self.loop = loop
        self.client = client
        self.semaphore = asyncio.Semaphore(concurrency)
        self.throttle_lock = asyncio.Lock()
        self.next_slot = 0.0


class _StepLog:
    """Stand-in for tqdm when output is a file: one line every `every` percent, no carriage returns."""

    def __init__(self, total: int, label: str, every: int = 10):
        self.total, self.label, self.every = total, label, every
        self.done, self.shown = 0, 0

    def update(self, n: int = 1) -> None:
        self.done += n
        percent = 100 * self.done // self.total
        if percent >= self.shown + self.every:
            self.shown = percent - percent % self.every
            print(f"    {self.label}: {self.shown:3d}% ({self.done}/{self.total})", flush=True)

    def close(self) -> None:
        pass


class BaseLLM:
    provider = "base"

    def __init__(self, model: str, *, system: Optional[str] = None, temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, concurrency: int = 16, rpm: Optional[float] = None,
                 max_retries: int = 6, timeout: float = 120.0, cache_dir: Optional[Union[str, Path]] = None,
                 execution: str = "", on_error: str = "raise", progress: bool = True, **default_kwargs: Any):
        if on_error not in ("raise", "return"):
            raise ValueError('on_error must be "raise" or "return"')
        self.model = model
        self.defaults = {"system": system, "temperature": temperature, "max_tokens": max_tokens, **default_kwargs}
        self.concurrency = concurrency
        self.rpm = rpm
        self.max_retries = max_retries
        self.timeout = timeout
        self.execution = execution
        self.on_error = on_error
        self.progress = progress
        self.cache = DiskCache(cache_dir) if cache_dir else None
        self._states: Dict[int, _LoopState] = {}
        self._dropped = set(self._unsupported_params())
        self._warned: set = set()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.provider}/{self.model})"

    def generate(self, prompts: Union[PromptLike, Sequence[PromptLike]], **kwargs: Any):
        return _BackgroundLoop.run(self.agenerate(prompts, **kwargs))

    def prompt(self, prompts: Union[PromptLike, Sequence[PromptLike]], **kwargs: Any):
        return _texts(self.generate(prompts, **kwargs))

    __call__ = prompt

    async def aprompt(self, prompts: Union[PromptLike, Sequence[PromptLike]], **kwargs: Any):
        return _texts(await self.agenerate(prompts, **kwargs))

    async def agenerate(self, prompts: Union[PromptLike, Sequence[PromptLike]], **kwargs: Any):
        single = _is_single(prompts)
        items = [prompts] if single else list(prompts)
        n = int(kwargs.pop("n", 1))
        if n < 1:
            raise ValueError("n must be >= 1")
        params = {**self.defaults, **{k: v for k, v in kwargs.items() if v is not None}}
        system = params.pop("system", None)
        conversations = [normalize_prompt(p, system) for p in items]

        results = await self._run_batch(conversations, params, n)

        failures = [g for row in results for g in row if not g.ok]
        if failures:
            message = (f"{self.provider}/{self.model}: {len(failures)}/{len(items) * n} requests failed; "
                       f"first error: {failures[0].error}")
            if self.on_error == "raise":
                raise LLMError(message)
            warnings.warn(message, stacklevel=2)

        shaped = [row[0] if n == 1 else row for row in results]
        return shaped[0] if single else shaped

    def _make_client(self) -> Any:
        raise NotImplementedError

    async def _generate_one(self, client: Any, system: Optional[str], messages: Conversation,
                            params: Dict[str, Any]) -> Generation:
        raise NotImplementedError

    def _classify_error(self, exc: BaseException) -> Tuple[bool, Optional[float]]:
        return isinstance(exc, (asyncio.TimeoutError, ConnectionError)), None

    def _unsupported_params(self) -> Sequence[str]:
        return ()

    async def _run_batch(self, conversations: List[Tuple[Optional[str], Conversation]],
                         params: Dict[str, Any], n: int) -> List[List[Generation]]:
        out: List[List[Optional[Generation]]] = [[None] * n for _ in conversations]
        bar = self._progress_bar(len(conversations) * n)

        async def run_one(i: int, sample: int) -> None:
            system, messages = conversations[i]
            out[i][sample] = await self._call(system, messages, params, sample)
            if bar is not None:
                bar.update(1)

        try:
            await asyncio.gather(*(run_one(i, s) for i in range(len(conversations)) for s in range(n)))
        finally:
            if bar is not None:
                bar.close()
        return out

    async def _call(self, system: Optional[str], messages: Conversation, params: Dict[str, Any],
                    sample: int) -> Generation:
        key = self.cache_key(system, messages, params, sample) if self.cache else None
        if key and (hit := self.cache.get(key)) is not None:
            return Generation(**hit, cached=True)

        state = self._state()
        error: Optional[BaseException] = None
        attempt = 0
        while True:
            effective = self.effective_params(params)
            retry_after = None
            async with state.semaphore:
                await self._throttle(state)
                try:
                    generation = await asyncio.wait_for(
                        self._generate_one(state.client, system, messages, effective), self.timeout)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    error = exc
                    retryable, retry_after = self._classify_error(exc)
                    if not retryable:
                        if self._drop_rejected_param(exc, effective):
                            continue
                        return self.failed(exc)
                else:
                    if key and generation.ok:
                        self.cache.set(key, generation.to_dict())
                    return generation
            if attempt >= self.max_retries:
                return self.failed(error)
            await asyncio.sleep(retry_after or min(60.0, 2.0 ** attempt) * random.uniform(0.5, 1.0))
            attempt += 1

    def effective_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        effective = {}
        for name, value in params.items():
            if value is None:
                continue
            if name in self._dropped:
                self._warn_once(name, f"{self.provider}/{self.model} does not accept `{name}`; ignoring it.")
                continue
            effective[name] = value
        return effective

    def _drop_rejected_param(self, exc: BaseException, params: Dict[str, Any]) -> bool:
        message = str(exc).lower()
        if not _UNSUPPORTED.search(message):
            return False
        for name, aliases in _PARAM_ALIASES.items():
            if name in params and any(alias in message for alias in aliases):
                self._dropped.add(name)
                return True
        return False

    def failed(self, exc: Optional[BaseException]) -> Generation:
        return Generation(text="", model=self.model, error=f"{type(exc).__name__}: {exc}")

    def _state(self) -> _LoopState:
        loop = asyncio.get_running_loop()
        state = self._states.get(id(loop))
        if state is None or state.loop is not loop:
            self._states = {k: v for k, v in self._states.items() if not v.loop.is_closed()}
            state = self._states[id(loop)] = _LoopState(loop, self._make_client(), self.concurrency)
        return state

    async def _throttle(self, state: _LoopState) -> None:
        if not self.rpm:
            return
        async with state.throttle_lock:
            now = time.monotonic()
            wait = state.next_slot - now
            state.next_slot = max(now, state.next_slot) + 60.0 / self.rpm
        if wait > 0:
            await asyncio.sleep(wait)

    def cache_key(self, system: Optional[str], messages: Conversation, params: Dict[str, Any], sample: int) -> str:
        sampled = (params.get("temperature") or 0) > 0
        payload = {"provider": self.provider, "model": self.model, "system": system, "messages": messages,
                   "params": params, "sample": sample, "execution": self.execution if sampled else ""}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def _progress_bar(self, total: int):
        if not self.progress or total < 2:
            return None
        if sys.stderr.isatty():
            return tqdm(total=total, desc=f"{self.provider}/{self.model}", leave=False)
        return _StepLog(total, f"{self.provider}/{self.model}")

    def _warn_once(self, tag: str, message: str) -> None:
        if tag not in self._warned:
            self._warned.add(tag)
            warnings.warn(message, stacklevel=3)


def normalize_prompt(prompt: PromptLike, system: Optional[str]) -> Tuple[Optional[str], Conversation]:
    if isinstance(prompt, str):
        return system, [{"role": "user", "content": prompt}]
    messages = [dict(m) for m in prompt]
    if messages and messages[0].get("role") == "system":
        if system is not None:
            raise ValueError("pass the system prompt as `system=` or as a system message, not both")
        system = messages.pop(0)["content"]
    if any(m.get("role") not in ("user", "assistant") for m in messages):
        raise ValueError("messages must be user/assistant turns after an optional leading system message")
    return system, messages


def retry_after_seconds(headers: Any) -> Optional[float]:
    try:
        value = headers.get("retry-after") if headers is not None else None
        return min(float(value), 120.0) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_single(prompts: Any) -> bool:
    return isinstance(prompts, str) or (isinstance(prompts, (list, tuple)) and bool(prompts)
                                        and isinstance(prompts[0], dict))


def _texts(result: Any) -> Any:
    return result.text if isinstance(result, Generation) else [_texts(r) for r in result]
