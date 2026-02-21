"""Tests for chat service."""

import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from harkd.chat.service import (
    MAX_TOOL_ITERATIONS,
    SUMMARIZE_THRESHOLD,
    ChatService,
    ChatStreamEvent,
)
from harkd.chat.tool_executor import ToolExecutor
from harkd.llm.client import LLMClient
from harkd.llm.types import LLMResponse, LLMStreamEvent
from harkd.storage.base import ChatThreadStorage
from harkd.storage.models import StorageChatMessage, StorageChatScope, StorageChatThread


async def make_stream_events(
    tokens: list[str],
    tool_calls: list[dict] | None = None,
) -> AsyncGenerator[LLMStreamEvent, None]:
    """Create an async generator yielding LLMStreamEvent objects.

    Args:
        tokens: list of text tokens to yield
        tool_calls: optional list of tool call dicts with id, name, args
    """
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            yield LLMStreamEvent(
                type="tool_call_start",
                tool_call_id=tc["id"],
                tool_name=tc["name"],
                tool_call_index=i,
            )
            args_str = json.dumps(tc.get("args", {}))
            yield LLMStreamEvent(
                type="tool_call_args",
                content=args_str,
                tool_call_index=i,
            )
            yield LLMStreamEvent(
                type="tool_call_end",
                tool_call_id=tc["id"],
                tool_name=tc["name"],
                content=args_str,
                tool_call_index=i,
            )

    for token in tokens:
        yield LLMStreamEvent(type="token", content=token)


@pytest.fixture
def mock_thread_storage():
    """Create mock ChatThreadStorage."""
    storage = AsyncMock(spec=ChatThreadStorage)
    storage.get.return_value = None
    storage.create.return_value = None
    storage.update.return_value = None
    return storage


@pytest.fixture
def mock_llm_client():
    """Create mock LLMClient."""
    client = MagicMock(spec=LLMClient)

    # Mock prompts property to return a mock PromptManager
    prompts_mock = MagicMock()

    def prompt_get(name):
        if name == "chat_summarize":
            return "Summarize: {conversation}"
        return "System prompt: {current_date}{scope_context}"

    prompts_mock.get = MagicMock(side_effect=prompt_get)
    type(client).prompts = PropertyMock(return_value=prompts_mock)

    # Default: stream returns text tokens
    async def default_stream(messages, tools=None):
        async for event in make_stream_events(["Hello", " world"]):
            yield event

    client.astream = MagicMock(side_effect=lambda *a, **kw: default_stream(*a, **kw))

    # Mock invoke for summarization
    client.invoke = AsyncMock(return_value=LLMResponse(content="Summary of conversation"))

    return client


@pytest.fixture
def mock_tool_executor():
    """Create mock ToolExecutor."""
    executor = AsyncMock(spec=ToolExecutor)
    executor.execute.return_value = json.dumps({"result": "ok"})
    return executor


@pytest.fixture
def service(mock_thread_storage, mock_llm_client, mock_tool_executor):
    """Create ChatService with mocked dependencies."""
    return ChatService(
        thread_storage=mock_thread_storage,
        llm_client=mock_llm_client,
        tool_executor=mock_tool_executor,
    )


async def collect_events(gen) -> list[ChatStreamEvent]:
    """Collect all events from an async generator."""
    events = []
    async for event in gen:
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_send_message_new_thread(service, mock_thread_storage):
    """Test that a new thread yields thread_created and persists via create()."""
    events = await collect_events(service.send_message("Hello", thread_id=None))

    types = [e.type for e in events]
    assert "thread_created" in types
    assert types[0] == "thread_created"
    mock_thread_storage.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_message_existing_thread(service, mock_thread_storage):
    """Test that an existing thread does not yield thread_created."""
    existing_thread = StorageChatThread(
        id="existing-thread",
        title="Existing",
        created_at=datetime(2026, 2, 10, 9, 0, 0, tzinfo=UTC),
        updated_at=datetime(2026, 2, 10, 9, 0, 0, tzinfo=UTC),
        scope=StorageChatScope(),
        messages=[],
    )
    mock_thread_storage.get.return_value = existing_thread

    events = await collect_events(service.send_message("Hello", thread_id="existing-thread"))

    types = [e.type for e in events]
    assert "thread_created" not in types
    mock_thread_storage.update.assert_awaited_once()
    mock_thread_storage.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_message_streams_tokens(service):
    """Test that token events match LLM output."""
    events = await collect_events(service.send_message("Hello"))

    token_events = [e for e in events if e.type == "token"]
    tokens = [e.data["content"] for e in token_events]
    assert tokens == ["Hello", " world"]


@pytest.mark.asyncio
async def test_send_message_tool_calling_loop(service, mock_llm_client, mock_tool_executor):
    """Test tool call round-trip: LLM calls tool, gets result, then responds."""
    call_count = 0

    async def stream_with_tool(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: return tool call
            async for event in make_stream_events(
                [],
                tool_calls=[
                    {"id": "tc-1", "name": "SearchRecordingsInput", "args": {"query": "standup"}}
                ],
            ):
                yield event
        else:
            # Second call: return text
            async for event in make_stream_events(["Found", " results"]):
                yield event

    mock_llm_client.astream = MagicMock(side_effect=lambda *a, **kw: stream_with_tool(*a, **kw))

    events = await collect_events(service.send_message("Find standups"))

    types = [e.type for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    assert "token" in types

    # Verify tool executor was called
    mock_tool_executor.execute.assert_awaited_once()
    call_args = mock_tool_executor.execute.call_args
    assert call_args[0][0] == "SearchRecordingsInput"


@pytest.mark.asyncio
async def test_send_message_max_tool_iterations(service, mock_llm_client, mock_tool_executor):
    """Test that the tool loop stops at MAX_TOOL_ITERATIONS."""

    async def always_tool_call(messages, tools=None):
        async for event in make_stream_events(
            [],
            tool_calls=[{"id": f"tc-{len(messages)}", "name": "SearchRecordingsInput", "args": {}}],
        ):
            yield event

    mock_llm_client.astream = MagicMock(side_effect=lambda *a, **kw: always_tool_call(*a, **kw))

    events = await collect_events(service.send_message("Loop forever"))

    tool_call_events = [e for e in events if e.type == "tool_call"]
    assert len(tool_call_events) == MAX_TOOL_ITERATIONS


@pytest.mark.asyncio
async def test_send_message_done_event(service):
    """Test that the last event is always 'done'."""
    events = await collect_events(service.send_message("Hello"))

    assert events[-1].type == "done"


@pytest.mark.asyncio
async def test_send_message_error_yields_error_event(service, mock_llm_client):
    """Test that LLM exception yields an error event."""

    async def raise_error(messages, tools=None):
        raise RuntimeError("LLM connection failed")
        # Make this an async generator
        yield  # noqa: E501  # pragma: no cover

    mock_llm_client.astream = MagicMock(side_effect=lambda *a, **kw: raise_error(*a, **kw))

    events = await collect_events(service.send_message("Hello"))

    types = [e.type for e in events]
    assert "error" in types
    error_event = next(e for e in events if e.type == "error")
    assert "LLM connection failed" in error_event.data["message"]


@pytest.mark.asyncio
async def test_thread_title_from_first_message(service, mock_thread_storage):
    """Test that first user message sets thread title."""
    await collect_events(service.send_message("What were the decisions in yesterday's standup?"))

    # The thread should have been created with a title from the message
    create_call = mock_thread_storage.create.call_args
    thread = create_call[0][0]
    assert thread.title == "What were the decisions in yesterday's standup?"


@pytest.mark.asyncio
async def test_context_summarization_triggered(service, mock_llm_client, mock_thread_storage):
    """Test that thread with >SUMMARIZE_THRESHOLD messages triggers summarization."""
    # Create thread with many messages
    existing_thread = StorageChatThread(
        id="long-thread",
        title="Long conversation",
        created_at=datetime(2026, 2, 10, 9, 0, 0, tzinfo=UTC),
        updated_at=datetime(2026, 2, 10, 9, 0, 0, tzinfo=UTC),
        scope=StorageChatScope(),
        messages=[
            StorageChatMessage(
                id=f"msg-{i}",
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
                timestamp=datetime(2026, 2, 10, 9, 0, i, tzinfo=UTC),
            )
            for i in range(SUMMARIZE_THRESHOLD)
        ],
    )
    mock_thread_storage.get.return_value = existing_thread

    await collect_events(service.send_message("One more message", thread_id="long-thread"))

    # After adding user msg + assistant msg, total > SUMMARIZE_THRESHOLD
    # The invoke() call should have been made for summarization
    mock_llm_client.invoke.assert_awaited()


@pytest.mark.asyncio
async def test_persist_failure_does_not_crash(service, mock_thread_storage):
    """Test that storage create() failure still yields done event."""
    mock_thread_storage.create.side_effect = RuntimeError("Disk full")

    events = await collect_events(service.send_message("Hello"))

    types = [e.type for e in events]
    assert "done" in types
