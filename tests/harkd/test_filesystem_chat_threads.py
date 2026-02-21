"""Tests for filesystem chat thread storage."""

from datetime import UTC, datetime

import pytest

from harkd.exceptions import ChatThreadNotFoundError, StorageError
from harkd.storage.filesystem.chat_threads import FilesystemChatThreadStorage
from harkd.storage.models import StorageChatMessage, StorageChatScope, StorageChatThread


@pytest.fixture
def temp_storage(tmp_path):
    """Create a temporary storage instance."""
    return FilesystemChatThreadStorage(tmp_path)


@pytest.fixture
def sample_thread():
    """Create a sample chat thread for testing."""
    return StorageChatThread(
        id="thread-abc-123",
        title="Weekly standup notes",
        created_at=datetime(2026, 2, 10, 9, 0, 0, tzinfo=UTC),
        updated_at=datetime(2026, 2, 10, 9, 5, 0, tzinfo=UTC),
        scope=StorageChatScope(speakers=["Alice", "Bob"]),
        messages=[
            StorageChatMessage(
                id="msg-1",
                role="user",
                content="Summarize the standup",
                timestamp=datetime(2026, 2, 10, 9, 0, 0, tzinfo=UTC),
            ),
            StorageChatMessage(
                id="msg-2",
                role="assistant",
                content="Here is a summary of the standup...",
                timestamp=datetime(2026, 2, 10, 9, 0, 5, tzinfo=UTC),
            ),
        ],
    )


@pytest.mark.asyncio
async def test_create_thread(temp_storage, sample_thread):
    """Test creating a new chat thread."""
    created = await temp_storage.create(sample_thread)

    assert created.id == sample_thread.id
    assert created.title == sample_thread.title

    # Verify file exists at chat-threads/{id}.json
    thread_file = temp_storage.threads_dir / f"{sample_thread.id}.json"
    assert thread_file.exists()


@pytest.mark.asyncio
async def test_create_duplicate_thread(temp_storage, sample_thread):
    """Test creating a thread with duplicate ID fails."""
    await temp_storage.create(sample_thread)

    with pytest.raises(StorageError) as exc_info:
        await temp_storage.create(sample_thread)

    assert "already exists" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_get_existing_thread(temp_storage, sample_thread):
    """Test getting an existing thread returns all fields."""
    await temp_storage.create(sample_thread)

    retrieved = await temp_storage.get(sample_thread.id)

    assert retrieved is not None
    assert retrieved.id == sample_thread.id
    assert retrieved.title == sample_thread.title
    assert retrieved.created_at == sample_thread.created_at
    assert retrieved.updated_at == sample_thread.updated_at
    assert retrieved.scope.speakers == ["Alice", "Bob"]
    assert len(retrieved.messages) == 2
    assert retrieved.messages[0].role == "user"
    assert retrieved.messages[1].role == "assistant"


@pytest.mark.asyncio
async def test_get_nonexistent_thread(temp_storage):
    """Test getting a non-existent thread returns None."""
    result = await temp_storage.get("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_list_empty(tmp_path):
    """Test listing when directory doesn't exist returns empty list."""
    storage = FilesystemChatThreadStorage(tmp_path / "nonexistent")
    threads = await storage.list()
    assert threads == []


@pytest.mark.asyncio
async def test_list_threads(temp_storage):
    """Test listing multiple threads sorted by updated_at desc."""
    t1 = StorageChatThread(
        id="thread-1",
        title="First",
        created_at=datetime(2026, 2, 10, 8, 0, 0, tzinfo=UTC),
        updated_at=datetime(2026, 2, 10, 8, 0, 0, tzinfo=UTC),
        scope=StorageChatScope(),
    )
    t2 = StorageChatThread(
        id="thread-2",
        title="Second",
        created_at=datetime(2026, 2, 10, 9, 0, 0, tzinfo=UTC),
        updated_at=datetime(2026, 2, 10, 11, 0, 0, tzinfo=UTC),
        scope=StorageChatScope(),
    )
    t3 = StorageChatThread(
        id="thread-3",
        title="Third",
        created_at=datetime(2026, 2, 10, 10, 0, 0, tzinfo=UTC),
        updated_at=datetime(2026, 2, 10, 10, 0, 0, tzinfo=UTC),
        scope=StorageChatScope(),
    )

    await temp_storage.create(t1)
    await temp_storage.create(t2)
    await temp_storage.create(t3)

    threads = await temp_storage.list()
    assert len(threads) == 3
    # Sorted by updated_at descending
    assert [t.id for t in threads] == ["thread-2", "thread-3", "thread-1"]


@pytest.mark.asyncio
async def test_list_strips_messages(temp_storage, sample_thread):
    """Test that listed threads have empty messages and None context_summary."""
    sample_thread.context_summary = "Some summary text"
    await temp_storage.create(sample_thread)

    threads = await temp_storage.list()
    assert len(threads) == 1
    assert threads[0].messages == []
    assert threads[0].context_summary is None


@pytest.mark.asyncio
async def test_update_thread(temp_storage, sample_thread):
    """Test updating a thread persists changes."""
    await temp_storage.create(sample_thread)

    sample_thread.title = "Updated title"
    updated = await temp_storage.update(sample_thread)
    assert updated.title == "Updated title"

    # Verify persistence
    retrieved = await temp_storage.get(sample_thread.id)
    assert retrieved.title == "Updated title"


@pytest.mark.asyncio
async def test_update_nonexistent(temp_storage, sample_thread):
    """Test updating a non-existent thread raises ChatThreadNotFoundError."""
    with pytest.raises(ChatThreadNotFoundError):
        await temp_storage.update(sample_thread)


@pytest.mark.asyncio
async def test_delete_thread(temp_storage, sample_thread):
    """Test deleting a thread removes the file."""
    await temp_storage.create(sample_thread)

    thread_file = temp_storage.threads_dir / f"{sample_thread.id}.json"
    assert thread_file.exists()

    await temp_storage.delete(sample_thread.id)

    assert not thread_file.exists()
    assert await temp_storage.get(sample_thread.id) is None


@pytest.mark.asyncio
async def test_delete_nonexistent(temp_storage):
    """Test deleting a non-existent thread raises ChatThreadNotFoundError."""
    with pytest.raises(ChatThreadNotFoundError):
        await temp_storage.delete("nonexistent-id")


@pytest.mark.asyncio
async def test_roundtrip_with_messages(temp_storage):
    """Test create/get roundtrip preserves messages with tool calls and citations."""
    thread = StorageChatThread(
        id="thread-roundtrip",
        title="Roundtrip test",
        created_at=datetime(2026, 2, 10, 9, 0, 0, tzinfo=UTC),
        updated_at=datetime(2026, 2, 10, 9, 5, 0, tzinfo=UTC),
        scope=StorageChatScope(recording_ids=["rec-1", "rec-2"]),
        messages=[
            StorageChatMessage(
                id="msg-1",
                role="user",
                content="What were the decisions?",
                timestamp=datetime(2026, 2, 10, 9, 0, 0, tzinfo=UTC),
            ),
            StorageChatMessage(
                id="msg-tool",
                role="tool",
                content='{"decisions": ["Use React"]}',
                timestamp=datetime(2026, 2, 10, 9, 0, 1, tzinfo=UTC),
                tool_call_id="call-1",
                tool_name="GetDecisionsInput",
            ),
            StorageChatMessage(
                id="msg-2",
                role="assistant",
                content="The team decided to use React.",
                timestamp=datetime(2026, 2, 10, 9, 0, 2, tzinfo=UTC),
                tool_calls=[{"id": "call-1", "name": "GetDecisionsInput", "args": {}}],
                citations=[{"recording_id": "rec-1", "recording_title": "Sprint Planning"}],
            ),
        ],
        context_summary="Earlier discussion about project setup.",
    )

    await temp_storage.create(thread)
    retrieved = await temp_storage.get("thread-roundtrip")

    assert retrieved is not None
    assert len(retrieved.messages) == 3
    assert retrieved.scope.recording_ids == ["rec-1", "rec-2"]
    assert retrieved.context_summary == "Earlier discussion about project setup."

    # Check tool message
    tool_msg = retrieved.messages[1]
    assert tool_msg.role == "tool"
    assert tool_msg.tool_call_id == "call-1"
    assert tool_msg.tool_name == "GetDecisionsInput"

    # Check assistant message with citations
    asst_msg = retrieved.messages[2]
    assert len(asst_msg.tool_calls) == 1
    assert asst_msg.tool_calls[0]["name"] == "GetDecisionsInput"
    assert len(asst_msg.citations) == 1
    assert asst_msg.citations[0]["recording_id"] == "rec-1"
