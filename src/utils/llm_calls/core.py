"""Provider-agnostic core shared by every backend.

Every backend subclasses `BaseLLM` and implements three hooks:
  - `_make_client()`                                   build the (async) SDK client
  - `_agenerate_one(client, system, messages, params)` one request -> `Generation`
  - `_classify_error(exc)`                             -> (retryable, retry_after_seconds)

Everything else — the call syntax, batching, bounded concurrency, rate limiting, retries with
exponential backoff + jitter, dropping sampling params a model rejects, on-disk caching, and the
sync/async bridge (works in scripts and inside Jupyter) — lives here, so it behaves identically
across providers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import sqlite3
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

Message = Dict[str, str]          # {"role": "user" | "assistant" | "system", "content": str}
Conversation = List[Message]
PromptLike = Union[str, Conversation]

class LLMError(RuntimeError):
    """Raised when requests still fail after retries (with `on_error="raise"`)."""


@dataclass
class Generation:
    text: str
    model: str
    finish_reason: Optional[str] = None
    logprobs: Optional[List[float]] = None   # per generated token (natural log), when supported
    tokens: Optional[List[str]] = None
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
        """Length-normalized sequence log-probability."""
        if not self.logprobs:
            return None
        return sum(self.logprobs) / len(self.logprobs)

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "text", "model", "finish_reason", "logprobs", "tokens",
            "input_tokens", "output_tokens", "error",
        )}

    def __str__(self) -> str:
        return self.text


# --------------------------------------------------------------------------------------------
# Sync bridge: one long-lived event loop on a daemon thread. Sync calls submit coroutines to it,
# so `generate()` works the same in plain scripts and inside an already-running (Jupyter) loop.
# --------------------------------------------------------------------------------------------
class _BackgroundLoop:
    _lock = threading.Lock()
    _loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def run(cls, coro):
        with cls._lock:
            if cls._loop is None:
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="llm_calls-loop", daemon=True).start()
                cls._loop = loop
        future = asyncio.run_coroutine_threadsafe(coro, cls._loop)
        try:
            return future.result()
        except KeyboardInterrupt:
            future.cancel()
            raise


class _DiskCache:
    """Tiny sqlite key-value store; only successful generations are written."""

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

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value)))
            self._db.commit()


class _LoopState:
    """Per-event-loop resources (SDK clients and asyncio primitives are loop-bound)."""

    def __init__(self, loop: asyncio.AbstractEventLoop, client: Any, concurrency: int):
        self.loop = loop
        self.client = client
        self.sem = asyncio.Semaphore(concurrency)
        self.throttle_lock = asyncio.Lock()
        self.next_slot = 0.0


_UNSUPPORTED_RE = re.compile(
    r"unsupported|not supported|does not support|doesn't support|not allowed|not permitted|"
    r"deprecated|only the default|is not available|unknown parameter|unrecognized|extra inputs"
)
_PARAM_ALIASES = {
    "temperature": ("temperature",),
    "top_p": ("top_p", "top-p", "topp"),
    "top_k": ("top_k", "top-k", "topk"),
    "logprobs": ("logprob",),
    "seed": ("seed",),
    "stop": ("stop",),
}


class BaseLLM:
    provider = "base"

    def __init__(
        self,
        model: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        concurrency: int = 16,
        rpm: Optional[float] = None,
        max_retries: int = 6,
        timeout: float = 120.0,
        cache_dir: Optional[Union[str, Path]] = None,
        on_error: str = "raise",
        progress: bool = True,
        **default_kwargs: Any,
    ):
        """
        Args:
            model: provider model id (e.g. "gpt-4o-mini", "claude-opus-5", "gemini-2.5-flash").
            system / temperature / max_tokens / **default_kwargs: per-instance defaults; any
                argument passed to `generate()` overrides them.
            concurrency: max in-flight requests.
            rpm: optional requests-per-minute cap (spaces request starts evenly).
            max_retries: retries on 408/409/429/5xx, timeouts and connection errors.
            timeout: per-request timeout in seconds.
            cache_dir: if set, successful generations are cached on disk keyed by
                (model, prompt, params, sample index) — reruns resume for free.
            on_error: "raise" -> raise LLMError after the whole batch finishes (successes are
                still cached); "return" -> failed items come back as Generation(error=...).
        """
        if on_error not in ("raise", "return"):
            raise ValueError('on_error must be "raise" or "return"')
        self.model = model
        self.defaults: Dict[str, Any] = {"system": system, "temperature": temperature,
                                         "max_tokens": max_tokens, **default_kwargs}
        self.concurrency = concurrency
        self.rpm = rpm
        self.max_retries = max_retries
        self.timeout = timeout
        self.on_error = on_error
        self.progress = progress
        self.cache = _DiskCache(cache_dir) if cache_dir else None
        self._states: Dict[int, _LoopState] = {}
        self._dropped: set = set(self._unsupported_params())
        self._warned: set = set()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(provider={self.provider!r}, model={self.model!r})"

    # ---------------------------------------------------------------- public API
    def generate(self, prompts: Union[PromptLike, Sequence[PromptLike]], **kwargs: Any):
        """Run prompt(s) and return `Generation` objects.

        `prompts` is a string, a chat conversation (list of {"role", "content"} dicts), or a list
        of either. Returns one `Generation` for a single prompt, a list for a batch. With `n > 1`
        each item becomes a list of `n` samples.

        Keyword args (all optional, same for every provider):
            system, temperature, max_tokens, top_p, top_k, stop, seed,
            logprobs (bool: return per-token logprobs where supported), n (samples per prompt),
            plus any provider-specific request fields, passed through verbatim.
        """
        return _BackgroundLoop.run(self.agenerate(prompts, **kwargs))

    def prompt(self, prompts: Union[PromptLike, Sequence[PromptLike]], **kwargs: Any):
        """Same as `generate` but returns only text (str / list[str] / nested lists for n > 1)."""
        return _texts(self.generate(prompts, **kwargs))

    __call__ = prompt

    async def agenerate(self, prompts: Union[PromptLike, Sequence[PromptLike]], **kwargs: Any):
        """Async version of `generate` (for use inside your own event loop)."""
        single = _is_single(prompts)
        items = [prompts] if single else list(prompts)
        n = int(kwargs.pop("n", 1))
        if n < 1:
            raise ValueError("n must be >= 1")
        params = {**self.defaults, **{k: v for k, v in kwargs.items() if v is not None}}
        system = params.pop("system", None)
        convs = [_normalize(p, system) for p in items]

        results = await self._run_batch(convs, params, n)

        failures = [g for row in results for g in row if not g.ok]
        if failures:
            msg = (f"{self.provider}/{self.model}: {len(failures)}/{len(items) * n} requests failed; "
                   f"first error: {failures[0].error}")
            if self.on_error == "raise":
                raise LLMError(msg)
            warnings.warn(msg, stacklevel=2)

        shaped = [row[0] if n == 1 else row for row in results]
        return shaped[0] if single else shaped

    async def aprompt(self, prompts: Union[PromptLike, Sequence[PromptLike]], **kwargs: Any):
        return _texts(await self.agenerate(prompts, **kwargs))

    # ---------------------------------------------------------------- provider hooks
    def _make_client(self) -> Any:
        raise NotImplementedError

    async def _agenerate_one(self, client: Any, system: Optional[str], messages: Conversation,
                             params: Dict[str, Any]) -> Generation:
        raise NotImplementedError

    def _classify_error(self, exc: BaseException) -> Tuple[bool, Optional[float]]:
        return isinstance(exc, (asyncio.TimeoutError, ConnectionError)), None

    def _unsupported_params(self) -> Sequence[str]:
        """Params known up front to be rejected by this model (dropped with a one-time warning)."""
        return ()

    # ---------------------------------------------------------------- machinery
    async def _run_batch(self, convs: List[Tuple[Optional[str], Conversation]],
                         params: Dict[str, Any], n: int) -> List[List[Generation]]:
        out: List[List[Optional[Generation]]] = [[None] * n for _ in convs]
        bar = self._bar(len(convs) * n)

        async def one(i: int, s: int) -> None:
            system, messages = convs[i]
            out[i][s] = await self._call(system, messages, params, s)
            if bar is not None:
                bar.update(1)

        try:
            await asyncio.gather(*(one(i, s) for i in range(len(convs)) for s in range(n)))
        finally:
            if bar is not None:
                bar.close()
        return out  # type: ignore[return-value]

    async def _call(self, system: Optional[str], messages: Conversation,
                    params: Dict[str, Any], sample: int) -> Generation:
        key = self._cache_key(system, messages, params, sample) if self.cache else None
        if key:
            hit = self.cache.get(key)
            if hit is not None:
                return Generation(**hit, cached=True)

        state = self._state()
        last_exc: Optional[BaseException] = None
        attempt = 0
        while True:
            effective = self._effective_params(params)
            retry_after = None
            async with state.sem:
                await self._throttle(state)
                try:
                    gen = await asyncio.wait_for(
                        self._agenerate_one(state.client, system, messages, effective), self.timeout)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 — classified below
                    last_exc = exc
                    retryable, retry_after = self._classify_error(exc)
                    if not retryable:
                        dropped = self._param_to_drop(exc, effective)
                        if dropped:
                            continue  # retry immediately without the rejected param
                        return self._failed(exc)
                else:
                    if key and gen.ok:
                        self.cache.set(key, gen.to_dict())
                    return gen
            if attempt >= self.max_retries:
                return self._failed(last_exc)
            delay = retry_after if retry_after else min(60.0, 2.0 ** attempt) * random.uniform(0.5, 1.0)
            attempt += 1
            await asyncio.sleep(delay)

    def _effective_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        eff = {}
        for k, v in params.items():
            if v is None:
                continue
            if k in self._dropped:
                self._warn_once(k, f"{self.provider}/{self.model} does not accept `{k}`; ignoring it.")
                continue
            eff[k] = v
        return eff

    def _param_to_drop(self, exc: BaseException, params: Dict[str, Any]) -> Optional[str]:
        msg = str(exc).lower()
        if not _UNSUPPORTED_RE.search(msg):
            return None
        for name, aliases in _PARAM_ALIASES.items():
            if name in params and any(a in msg for a in aliases):
                self._dropped.add(name)
                return name
        return None

    def _failed(self, exc: Optional[BaseException]) -> Generation:
        return Generation(text="", model=self.model, error=f"{type(exc).__name__}: {exc}")

    def _state(self) -> _LoopState:
        loop = asyncio.get_running_loop()
        state = self._states.get(id(loop))
        if state is None or state.loop is not loop:  # ids of closed loops get reused
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

    def _cache_key(self, system, messages, params, sample) -> str:
        blob = json.dumps({"provider": self.provider, "model": self.model, "system": system,
                           "messages": messages, "params": params, "sample": sample},
                          sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _bar(self, total: int):
        if not self.progress or tqdm is None or total < 2:
            return None
        return tqdm(total=total, desc=f"{self.provider}/{self.model}", leave=False)

    def _warn_once(self, tag: str, msg: str) -> None:
        if tag not in self._warned:
            self._warned.add(tag)
            warnings.warn(msg, stacklevel=3)


# ------------------------------------------------------------------------------ helpers
def _is_single(prompts: Any) -> bool:
    if isinstance(prompts, str):
        return True
    if isinstance(prompts, (list, tuple)) and prompts and isinstance(prompts[0], dict):
        return True  # one conversation
    return False


def _normalize(prompt: PromptLike, system: Optional[str]) -> Tuple[Optional[str], Conversation]:
    if isinstance(prompt, str):
        return system, [{"role": "user", "content": prompt}]
    messages = [dict(m) for m in prompt]
    if messages and messages[0].get("role") == "system":
        if system is not None:
            raise ValueError("Pass the system prompt either as `system=` or as a system message, not both.")
        system = messages.pop(0)["content"]
    for m in messages:
        if m.get("role") not in ("user", "assistant"):
            raise ValueError(f"Unsupported message role {m.get('role')!r} (use user/assistant, "
                             "with an optional leading system message).")
    return system, messages


def _texts(result: Any) -> Any:
    if isinstance(result, Generation):
        return result.text
    return [_texts(r) for r in result]


def retry_after_seconds(headers: Any) -> Optional[float]:
    """Parse a `retry-after` header (seconds form) if present."""
    try:
        value = headers.get("retry-after") if headers is not None else None
        return min(float(value), 120.0) if value is not None else None
    except (TypeError, ValueError):
        return None
