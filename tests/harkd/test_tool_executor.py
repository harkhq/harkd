"""Tests for chat tool executor."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from harkd.chat.tool_executor import ToolExecutor
from harkd.storage.base import RecordingStorage
from harkd.storage.models import StorageChatScope, StorageRecording


def _make_recording(
    id: str,
    title: str,
    created_at: datetime,
    speakers: list[str] | None = None,
    tags: list[str] | None = None,
    transcript: str | None = None,
    segments: list[dict] | None = None,
    tasks: list[dict] | None = None,
    decisions: list[str] | None = None,
    executive_summary: list[str] | None = None,
    meeting_notes: list[dict] | None = None,
) -> StorageRecording:
    return StorageRecording(
        id=id,
        status="complete",
        created_at=created_at,
        title=title,
        duration=60.0,
        speakers=list(speakers) if speakers else [],
        tags=list(tags) if tags else [],
        transcript=transcript,
        segments=list(segments) if segments else [],
        tasks=list(tasks) if tasks else [],
        decisions=list(decisions) if decisions else [],
        executive_summary=list(executive_summary) if executive_summary else [],
        meeting_notes=list(meeting_notes) if meeting_notes else [],
        settings={},
    )


@pytest.fixture
def sample_recordings():
    """Create sample recordings for testing."""
    return [
        _make_recording(
            id="rec-1",
            title="Sprint Planning",
            created_at=datetime(2026, 2, 1, 10, 0, 0, tzinfo=UTC),
            speakers=["Alice", "Bob"],
            tags=["sprint", "planning"],
            transcript="Alice discussed the new feature. Bob raised concerns about timeline.",
            segments=[
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Alice discussed the new feature.",
                    "speaker": "Alice",
                },
                {
                    "start": 5.0,
                    "end": 10.0,
                    "text": "Bob raised concerns about timeline.",
                    "speaker": "Bob",
                },
                {
                    "start": 10.0,
                    "end": 15.0,
                    "text": "The team agreed to proceed.",
                    "speaker": "Alice",
                },
            ],
            tasks=[
                {"description": "Write RFC", "assignee": "Alice", "due": "2026-02-05"},
                {"description": "Review timeline", "assignee": "Bob"},
            ],
            decisions=["Proceed with new feature", "Two-week sprint cycle"],
        ),
        _make_recording(
            id="rec-2",
            title="Design Review",
            created_at=datetime(2026, 2, 5, 14, 0, 0, tzinfo=UTC),
            speakers=["Alice", "Charlie"],
            tags=["design"],
            transcript="Charlie presented the UI mockups. Alice gave feedback.",
            segments=[
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Charlie presented the UI mockups.",
                    "speaker": "Charlie",
                },
                {
                    "start": 5.0,
                    "end": 10.0,
                    "text": "Alice gave feedback.",
                    "speaker": "Alice",
                },
            ],
            tasks=[
                {"description": "Update mockups", "assignee": "Charlie"},
            ],
            decisions=["Use React for frontend"],
        ),
        _make_recording(
            id="rec-3",
            title="Retrospective",
            created_at=datetime(2026, 2, 10, 16, 0, 0, tzinfo=UTC),
            speakers=["Bob", "Charlie"],
            tags=["retro"],
            transcript="Bob shared what went well. Charlie discussed improvements.",
            segments=[
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Bob shared what went well.",
                    "speaker": "Bob",
                },
                {
                    "start": 5.0,
                    "end": 10.0,
                    "text": "Charlie discussed improvements.",
                    "speaker": "Charlie",
                },
            ],
            tasks=[],
            decisions=["Adopt daily standups"],
        ),
    ]


@pytest.fixture
def mock_storage(sample_recordings):
    """Create a mock RecordingStorage."""
    storage = AsyncMock(spec=RecordingStorage)
    storage.list.return_value = sample_recordings

    async def mock_get(recording_id):
        for r in sample_recordings:
            if r.id == recording_id:
                return r
        return None

    storage.get.side_effect = mock_get
    return storage


@pytest.fixture
def executor(mock_storage):
    """Create a ToolExecutor with mock storage."""
    return ToolExecutor(mock_storage)


@pytest.mark.asyncio
async def test_search_recordings_no_filters(executor):
    """Test search with no filters returns all complete recordings."""
    result_str = await executor.execute("SearchRecordingsInput", {})
    result = json.loads(result_str)

    assert len(result) == 3


@pytest.mark.asyncio
async def test_search_recordings_by_query(executor):
    """Test search with keyword match on title."""
    result_str = await executor.execute("SearchRecordingsInput", {"query": "planning"})
    result = json.loads(result_str)

    assert len(result) == 1
    assert result[0]["id"] == "rec-1"


@pytest.mark.asyncio
async def test_search_recordings_by_date_range(executor):
    """Test search with date_from/date_to filtering."""
    result_str = await executor.execute(
        "SearchRecordingsInput",
        {"date_from": "2026-02-04", "date_to": "2026-02-06"},
    )
    result = json.loads(result_str)

    assert len(result) == 1
    assert result[0]["id"] == "rec-2"


@pytest.mark.asyncio
async def test_search_recordings_by_speakers(executor):
    """Test search filtered by speakers."""
    result_str = await executor.execute(
        "SearchRecordingsInput",
        {"speakers": ["Charlie"]},
    )
    result = json.loads(result_str)

    assert len(result) == 2
    ids = {r["id"] for r in result}
    assert ids == {"rec-2", "rec-3"}


@pytest.mark.asyncio
async def test_search_recordings_by_tags(executor):
    """Test search filtered by tags."""
    result_str = await executor.execute(
        "SearchRecordingsInput",
        {"tags": ["design"]},
    )
    result = json.loads(result_str)

    assert len(result) == 1
    assert result[0]["id"] == "rec-2"


@pytest.mark.asyncio
async def test_search_recordings_limit(executor):
    """Test search respects limit param."""
    result_str = await executor.execute(
        "SearchRecordingsInput",
        {"limit": 2},
    )
    result = json.loads(result_str)

    assert len(result) == 2


@pytest.mark.asyncio
async def test_search_recordings_with_scope(executor):
    """Test search with scope constrains results."""
    scope = StorageChatScope(recording_ids=["rec-1", "rec-2"])
    result_str = await executor.execute("SearchRecordingsInput", {}, scope=scope)
    result = json.loads(result_str)

    assert len(result) == 2
    ids = {r["id"] for r in result}
    assert ids == {"rec-1", "rec-2"}


@pytest.mark.asyncio
async def test_get_recording_detail(executor):
    """Test getting full recording detail with truncated transcript."""
    result_str = await executor.execute(
        "GetRecordingDetailInput",
        {"recording_id": "rec-1"},
    )
    result = json.loads(result_str)

    assert result["id"] == "rec-1"
    assert result["title"] == "Sprint Planning"
    assert result["speakers"] == ["Alice", "Bob"]
    assert "transcript_preview" in result
    assert len(result["transcript_preview"]) <= 3000


@pytest.mark.asyncio
async def test_get_recording_detail_not_found(executor):
    """Test getting a non-existent recording returns error."""
    result_str = await executor.execute(
        "GetRecordingDetailInput",
        {"recording_id": "nonexistent"},
    )
    result = json.loads(result_str)

    assert "error" in result


@pytest.mark.asyncio
async def test_get_recording_detail_scope_denied(executor):
    """Test that recording outside scope returns error."""
    scope = StorageChatScope(recording_ids=["rec-2"])
    result_str = await executor.execute(
        "GetRecordingDetailInput",
        {"recording_id": "rec-1"},
        scope=scope,
    )
    result = json.loads(result_str)

    assert "error" in result
    assert "outside the current scope" in result["error"]


@pytest.mark.asyncio
async def test_get_transcript_segments(executor):
    """Test keyword search returns matching segments with context."""
    result_str = await executor.execute(
        "GetTranscriptSegmentsInput",
        {"query": "concerns"},
    )
    result = json.loads(result_str)

    assert len(result) >= 1
    match = result[0]
    assert match["recording_id"] == "rec-1"
    assert "concerns" in match["text"].lower()
    # Context should include surrounding segments
    assert len(match["context"]) >= 2


@pytest.mark.asyncio
async def test_get_transcript_segments_empty_query(executor):
    """Test keyword search with no matches returns empty."""
    result_str = await executor.execute(
        "GetTranscriptSegmentsInput",
        {"query": "xyznonexistent"},
    )
    result = json.loads(result_str)

    assert result == []


@pytest.mark.asyncio
async def test_get_tasks_all(executor):
    """Test aggregating tasks across recordings."""
    result_str = await executor.execute("GetTasksInput", {})
    result = json.loads(result_str)

    assert len(result) == 3
    descriptions = {t["description"] for t in result}
    assert "Write RFC" in descriptions
    assert "Update mockups" in descriptions


@pytest.mark.asyncio
async def test_get_tasks_by_assignee(executor):
    """Test filtering tasks by assignee."""
    result_str = await executor.execute(
        "GetTasksInput",
        {"assignee": "Alice"},
    )
    result = json.loads(result_str)

    assert len(result) == 1
    assert result[0]["assignee"] == "Alice"


@pytest.mark.asyncio
async def test_get_decisions_with_date_range(executor):
    """Test getting decisions filtered by date range."""
    result_str = await executor.execute(
        "GetDecisionsInput",
        {"date_from": "2026-02-04", "date_to": "2026-02-06"},
    )
    result = json.loads(result_str)

    assert len(result) == 1
    assert result[0]["decision"] == "Use React for frontend"


@pytest.mark.asyncio
async def test_execute_unknown_tool(executor):
    """Test executing an unknown tool returns error."""
    result_str = await executor.execute("UnknownTool", {})
    result = json.loads(result_str)

    assert "error" in result
    assert "Unknown tool: UnknownTool" in result["error"]


@pytest.mark.asyncio
async def test_execute_exception_handling(mock_storage):
    """Test that storage exceptions are caught and returned as JSON error."""
    mock_storage.list.side_effect = RuntimeError("Connection failed")
    executor = ToolExecutor(mock_storage)

    result_str = await executor.execute("SearchRecordingsInput", {})
    result = json.loads(result_str)

    assert "error" in result
    assert "Connection failed" in result["error"]


@pytest.mark.asyncio
async def test_apply_scope_recording_ids(executor, sample_recordings):
    """Test _apply_scope filters by specific recording IDs."""
    scope = StorageChatScope(recording_ids=["rec-1", "rec-3"])
    filtered = ToolExecutor._apply_scope(sample_recordings, scope)

    assert len(filtered) == 2
    ids = {r.id for r in filtered}
    assert ids == {"rec-1", "rec-3"}


@pytest.mark.asyncio
async def test_apply_scope_date_range(executor, sample_recordings):
    """Test _apply_scope with datetime vs date edge cases."""
    # Use datetime scope (as stored in StorageChatScope)
    scope = StorageChatScope(
        date_from=datetime(2026, 2, 4, 0, 0, 0, tzinfo=UTC),
        date_to=datetime(2026, 2, 6, 23, 59, 59, tzinfo=UTC),
    )
    filtered = ToolExecutor._apply_scope(sample_recordings, scope)

    assert len(filtered) == 1
    assert filtered[0].id == "rec-2"


@pytest.mark.asyncio
async def test_apply_scope_speakers_case_insensitive(executor, sample_recordings):
    """Test that speaker scope filtering is case insensitive."""
    scope = StorageChatScope(speakers=["alice"])
    filtered = ToolExecutor._apply_scope(sample_recordings, scope)

    # rec-1 and rec-2 have Alice as a speaker
    assert len(filtered) == 2
    ids = {r.id for r in filtered}
    assert ids == {"rec-1", "rec-2"}
