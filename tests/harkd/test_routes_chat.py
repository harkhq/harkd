"""Tests for chat API routes."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from harkd.api.app import create_app
from harkd.api.deps import get_chat_service
from harkd.chat.service import ChatService
from harkd.chat.tool_executor import ToolExecutor
from harkd.config import HarkdSettings, LLMSettings, StorageSettings
from harkd.storage.filesystem.chat_threads import FilesystemChatThreadStorage
from harkd.storage.models import StorageChatMessage, StorageChatScope, StorageChatThread


@pytest.fixture(scope="function")
def settings(tmp_path_factory):
    """Create test settings with temp storage and LLM enabled."""
    tmp_dir = tmp_path_factory.mktemp("chat")
    return HarkdSettings(
        storage=StorageSettings(base_path=tmp_dir),
        llm=LLMSettings(enabled=True, provider="openai", api_key="test-key"),
    )


@pytest.fixture(scope="function")
def settings_no_llm(tmp_path_factory):
    """Create test settings with LLM disabled."""
    tmp_dir = tmp_path_factory.mktemp("chat-no-llm")
    return HarkdSettings(
        storage=StorageSettings(base_path=tmp_dir),
        llm=LLMSettings(enabled=False),
    )


@pytest.fixture
def client(settings):
    """Create test client with LLM enabled, using mock LLM client."""
    from harkd.api import deps
    from harkd.config import get_settings

    deps._recording_services.clear()
    deps._voice_profile_services.clear()
    deps._processing_workers.clear()
    deps._chat_services.clear()

    base_path = settings.storage.base_path
    base_path.mkdir(parents=True, exist_ok=True)

    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings

    # Override get_chat_service to avoid creating a real LLMClient
    # (which triggers heavy torch/transformers imports)
    thread_storage = FilesystemChatThreadStorage(base_path)
    mock_llm = MagicMock()
    mock_tool_executor = AsyncMock(spec=ToolExecutor)
    chat_service = ChatService(
        thread_storage=thread_storage,
        llm_client=mock_llm,
        tool_executor=mock_tool_executor,
    )
    app.dependency_overrides[get_chat_service] = lambda: chat_service

    return TestClient(app)


@pytest.fixture
def client_no_llm(settings_no_llm):
    """Create test client with LLM disabled."""
    from harkd.api import deps
    from harkd.config import get_settings

    deps._recording_services.clear()
    deps._voice_profile_services.clear()
    deps._processing_workers.clear()
    deps._chat_services.clear()

    base_path = settings_no_llm.storage.base_path
    base_path.mkdir(parents=True, exist_ok=True)

    app = create_app(settings_no_llm)
    app.dependency_overrides[get_settings] = lambda: settings_no_llm

    return TestClient(app)


def create_thread_in_storage(settings, thread: StorageChatThread):
    """Helper to write a chat thread directly to filesystem."""
    threads_dir = settings.storage.base_path / "chat-threads"
    threads_dir.mkdir(parents=True, exist_ok=True)

    thread_file = threads_dir / f"{thread.id}.json"
    data = thread.model_dump(mode="json")
    with open(thread_file, "w") as f:
        json.dump(data, f, indent=2)


def _make_thread(
    id: str,
    title: str = "Test thread",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> StorageChatThread:
    now = datetime(2026, 2, 10, 9, 0, 0, tzinfo=UTC)
    return StorageChatThread(
        id=id,
        title=title,
        created_at=created_at or now,
        updated_at=updated_at or now,
        scope=StorageChatScope(),
        messages=[
            StorageChatMessage(
                id="msg-1",
                role="user",
                content="Hello",
                timestamp=now,
            ),
            StorageChatMessage(
                id="msg-2",
                role="assistant",
                content="Hi there!",
                timestamp=now,
            ),
        ],
    )


class TestListThreads:
    """Tests for GET /api/v1/chat/threads."""

    def test_list_threads_empty(self, client):
        """Test listing when no threads exist."""
        response = client.get("/api/v1/chat/threads")

        assert response.status_code == 200
        data = response.json()
        assert data == {"threads": [], "total": 0}

    def test_list_threads(self, client, settings):
        """Test listing threads returns all threads."""
        t1 = _make_thread("thread-1", "First thread")
        t2 = _make_thread(
            "thread-2",
            "Second thread",
            updated_at=datetime(2026, 2, 11, 9, 0, 0, tzinfo=UTC),
        )
        create_thread_in_storage(settings, t1)
        create_thread_in_storage(settings, t2)

        response = client.get("/api/v1/chat/threads")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["threads"]) == 2

        titles = {t["title"] for t in data["threads"]}
        assert "First thread" in titles
        assert "Second thread" in titles


class TestGetThread:
    """Tests for GET /api/v1/chat/threads/{id}."""

    def test_get_thread(self, client, settings):
        """Test getting a thread returns full data including messages."""
        thread = _make_thread("thread-abc", "My conversation")
        create_thread_in_storage(settings, thread)

        response = client.get("/api/v1/chat/threads/thread-abc")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "thread-abc"
        assert data["title"] == "My conversation"
        # Messages should be present (only user/assistant, not tool)
        assert len(data["messages"]) == 2

    def test_get_thread_not_found(self, client):
        """Test 404 for non-existent thread."""
        response = client.get("/api/v1/chat/threads/nonexistent")

        assert response.status_code == 404
        error = response.json()
        assert error["error"]["code"] == "CHAT_THREAD_NOT_FOUND"


class TestUpdateThread:
    """Tests for PATCH /api/v1/chat/threads/{id}."""

    def test_update_thread_title(self, client, settings):
        """Test updating thread title."""
        thread = _make_thread("thread-upd", "Old title")
        create_thread_in_storage(settings, thread)

        response = client.patch(
            "/api/v1/chat/threads/thread-upd",
            json={"title": "New title"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "New title"

        # Verify persistence
        response = client.get("/api/v1/chat/threads/thread-upd")
        assert response.json()["title"] == "New title"

    def test_update_thread_not_found(self, client):
        """Test 404 for updating non-existent thread."""
        response = client.patch(
            "/api/v1/chat/threads/nonexistent",
            json={"title": "New title"},
        )

        assert response.status_code == 404


class TestDeleteThread:
    """Tests for DELETE /api/v1/chat/threads/{id}."""

    def test_delete_thread(self, client, settings):
        """Test deleting a thread removes it."""
        thread = _make_thread("thread-del", "Delete me")
        create_thread_in_storage(settings, thread)

        # Verify it exists
        response = client.get("/api/v1/chat/threads/thread-del")
        assert response.status_code == 200

        # Delete it
        response = client.delete("/api/v1/chat/threads/thread-del")
        assert response.status_code == 204

        # Verify file is removed
        thread_file = settings.storage.base_path / "chat-threads" / "thread-del.json"
        assert not thread_file.exists()

    def test_delete_thread_not_found(self, client):
        """Test 404 for deleting non-existent thread."""
        response = client.delete("/api/v1/chat/threads/nonexistent")

        assert response.status_code == 404


class TestChatSendGuard:
    """Tests for POST /api/v1/chat/send requiring LLM."""

    def test_chat_send_requires_llm_enabled(self, client_no_llm):
        """Test that chat/send returns 503 when LLM is disabled."""
        response = client_no_llm.post(
            "/api/v1/chat/send",
            json={"message": "Hello"},
        )

        assert response.status_code == 503
        error = response.json()
        assert error["error"]["code"] == "LLM_NOT_CONFIGURED"
