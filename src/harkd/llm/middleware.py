"""Middleware chain for LLM calls."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from harkd.llm.types import LLMResponse, TokenUsage

logger = logging.getLogger(__name__)


class Middleware(ABC):
    """Base class for LLM middleware."""

    @abstractmethod
    async def __call__(self, request: dict[str, Any], next_fn: Callable[..., Any]) -> LLMResponse:
        """Process request through middleware.

        Args:
            request: The request dict (messages, tools, etc.)
            next_fn: The next middleware or final LLM call

        Returns:
            LLM response
        """


class LoggingMiddleware(Middleware):
    """Logs LLM request/response details."""

    async def __call__(self, request: dict[str, Any], next_fn: Callable[..., Any]) -> LLMResponse:
        msg_count = len(request.get("messages", []))
        logger.info(f"LLM request: {msg_count} messages")
        start = time.monotonic()

        response = await next_fn(request)

        elapsed = time.monotonic() - start
        usage_str = ""
        if response.usage:
            usage_str = f", tokens={response.usage.total_tokens}"
        logger.info(
            f"LLM response: {len(response.content)} chars, "
            f"{elapsed:.2f}s{usage_str}, cached={response.cached}"
        )
        return response


class TokenStatsMiddleware(Middleware):
    """Tracks cumulative token usage across calls."""

    def __init__(self):
        self._total_usage = TokenUsage()

    @property
    def total_usage(self) -> TokenUsage:
        """Get cumulative token usage."""
        return self._total_usage

    async def __call__(self, request: dict[str, Any], next_fn: Callable[..., Any]) -> LLMResponse:
        response = await next_fn(request)

        if response.usage:
            self._total_usage.prompt_tokens += response.usage.prompt_tokens
            self._total_usage.completion_tokens += response.usage.completion_tokens
            self._total_usage.total_tokens += response.usage.total_tokens

        return response


class CacheMiddleware(Middleware):
    """Simple in-memory hash-based response cache."""

    def __init__(self, ttl_seconds: int = 3600):
        self._cache: dict[str, tuple[float, LLMResponse]] = {}
        self._ttl = ttl_seconds

    async def __call__(self, request: dict[str, Any], next_fn: Callable[..., Any]) -> LLMResponse:
        cache_key = self._make_key(request)

        # Check cache
        if cache_key in self._cache:
            cached_at, cached_response = self._cache[cache_key]
            if time.monotonic() - cached_at < self._ttl:
                return LLMResponse(
                    content=cached_response.content,
                    usage=cached_response.usage,
                    model=cached_response.model,
                    cached=True,
                )
            # Expired
            del self._cache[cache_key]

        response = await next_fn(request)

        # Store in cache
        self._cache[cache_key] = (time.monotonic(), response)

        return response

    @staticmethod
    def _make_key(request: dict[str, Any]) -> str:
        """Create a deterministic cache key from the request."""
        serialized = json.dumps(request, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()


def build_middleware_chain(
    middlewares: list[Middleware], final_fn: Callable[..., Any]
) -> Callable[..., Any]:
    """Build a middleware chain ending with the final function.

    Execution order: first middleware in list wraps the second, etc.
    E.g. [cache, logging, stats] -> cache(logging(stats(final_fn)))

    Args:
        middlewares: List of middleware instances
        final_fn: The final async function to call

    Returns:
        An async callable that runs through the full chain
    """
    chain = final_fn
    for mw in reversed(middlewares):

        def _wrap(middleware=mw, next_fn=chain):
            async def wrapped(request: dict[str, Any]) -> LLMResponse:
                return await middleware(request, next_fn)

            return wrapped

        chain = _wrap()

    return chain
