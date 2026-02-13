"""Tests for LLM middleware chain."""

import logging

import pytest

from harkd.llm.middleware import (
    CacheMiddleware,
    LoggingMiddleware,
    TokenStatsMiddleware,
    build_middleware_chain,
)
from harkd.llm.types import LLMResponse, TokenUsage


@pytest.fixture
def sample_response():
    """Create a sample LLM response."""
    return LLMResponse(
        content="Test response",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        model="test-model",
    )


@pytest.fixture
def sample_request():
    """Create a sample request dict."""
    return {"messages": [{"role": "user", "content": "Hello"}]}


class TestLoggingMiddleware:
    """Tests for LoggingMiddleware."""

    @pytest.mark.asyncio
    async def test_logs_request_and_response(self, sample_request, sample_response, caplog):
        """Test that logging middleware logs properly."""
        mw = LoggingMiddleware()

        async def next_fn(request):
            return sample_response

        with caplog.at_level(logging.INFO, logger="harkd.llm.middleware"):
            result = await mw(sample_request, next_fn)

        assert result.content == "Test response"
        assert "LLM request: 1 messages" in caplog.text
        assert "LLM response:" in caplog.text
        assert "tokens=30" in caplog.text

    @pytest.mark.asyncio
    async def test_logs_without_usage(self, sample_request, caplog):
        """Test logging when response has no usage info."""
        mw = LoggingMiddleware()
        response = LLMResponse(content="No usage")

        async def next_fn(request):
            return response

        with caplog.at_level(logging.INFO, logger="harkd.llm.middleware"):
            result = await mw(sample_request, next_fn)

        assert result.content == "No usage"


class TestTokenStatsMiddleware:
    """Tests for TokenStatsMiddleware."""

    @pytest.mark.asyncio
    async def test_accumulates_usage(self, sample_request):
        """Test that token stats accumulate across calls."""
        mw = TokenStatsMiddleware()

        response1 = LLMResponse(
            content="r1",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        response2 = LLMResponse(
            content="r2",
            usage=TokenUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40),
        )

        call_count = 0

        async def next_fn(request):
            nonlocal call_count
            call_count += 1
            return response1 if call_count == 1 else response2

        await mw(sample_request, next_fn)
        await mw(sample_request, next_fn)

        assert mw.total_usage.prompt_tokens == 25
        assert mw.total_usage.completion_tokens == 45
        assert mw.total_usage.total_tokens == 70

    @pytest.mark.asyncio
    async def test_no_usage_in_response(self, sample_request):
        """Test handling of responses without usage info."""
        mw = TokenStatsMiddleware()

        async def next_fn(request):
            return LLMResponse(content="no usage")

        await mw(sample_request, next_fn)

        assert mw.total_usage.total_tokens == 0


class TestCacheMiddleware:
    """Tests for CacheMiddleware."""

    @pytest.mark.asyncio
    async def test_returns_cached_response(self, sample_request, sample_response):
        """Test that duplicate request returns cached response."""
        mw = CacheMiddleware()
        call_count = 0

        async def next_fn(request):
            nonlocal call_count
            call_count += 1
            return sample_response

        # First call - cache miss
        r1 = await mw(sample_request, next_fn)
        assert r1.content == "Test response"
        assert r1.cached is False
        assert call_count == 1

        # Second call - cache hit
        r2 = await mw(sample_request, next_fn)
        assert r2.content == "Test response"
        assert r2.cached is True
        assert call_count == 1  # next_fn not called again

    @pytest.mark.asyncio
    async def test_different_requests_not_cached(self, sample_response):
        """Test that different requests aren't served from cache."""
        mw = CacheMiddleware()
        call_count = 0

        async def next_fn(request):
            nonlocal call_count
            call_count += 1
            return sample_response

        req1 = {"messages": [{"role": "user", "content": "Hello"}]}
        req2 = {"messages": [{"role": "user", "content": "World"}]}

        await mw(req1, next_fn)
        await mw(req2, next_fn)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_expired_cache_refetches(self, sample_request, sample_response):
        """Test that expired cache entries are refetched."""
        mw = CacheMiddleware(ttl_seconds=0)  # Immediate expiry
        call_count = 0

        async def next_fn(request):
            nonlocal call_count
            call_count += 1
            return sample_response

        await mw(sample_request, next_fn)
        await mw(sample_request, next_fn)

        assert call_count == 2  # Both calls hit next_fn


class TestBuildMiddlewareChain:
    """Tests for middleware chain building."""

    @pytest.mark.asyncio
    async def test_chain_execution_order(self, sample_request):
        """Test that middleware executes in correct order."""
        order = []

        class TrackingMiddleware1(LoggingMiddleware):
            async def __call__(self, request, next_fn):
                order.append("m1_before")
                result = await next_fn(request)
                order.append("m1_after")
                return result

        class TrackingMiddleware2(LoggingMiddleware):
            async def __call__(self, request, next_fn):
                order.append("m2_before")
                result = await next_fn(request)
                order.append("m2_after")
                return result

        async def final_fn(request):
            order.append("final")
            return LLMResponse(content="done")

        chain = build_middleware_chain(
            [TrackingMiddleware1(), TrackingMiddleware2()],
            final_fn,
        )

        result = await chain(sample_request)
        assert result.content == "done"
        assert order == ["m1_before", "m2_before", "final", "m2_after", "m1_after"]

    @pytest.mark.asyncio
    async def test_empty_middleware_calls_final(self, sample_request):
        """Test chain with no middleware calls final directly."""

        async def final_fn(request):
            return LLMResponse(content="direct")

        chain = build_middleware_chain([], final_fn)
        result = await chain(sample_request)
        assert result.content == "direct"

    @pytest.mark.asyncio
    async def test_error_propagates_through_chain(self, sample_request):
        """Test that exceptions from final_fn propagate through middleware."""

        async def failing_fn(request):
            raise RuntimeError("LLM call failed")

        chain = build_middleware_chain(
            [LoggingMiddleware(), TokenStatsMiddleware()],
            failing_fn,
        )

        with pytest.raises(RuntimeError, match="LLM call failed"):
            await chain(sample_request)
