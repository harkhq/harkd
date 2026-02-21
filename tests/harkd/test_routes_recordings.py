"""Tests for recordings API routes."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from harkd.api.app import create_app
from harkd.api.models.recording import RecordingStatus
from harkd.config import HarkdSettings, StorageSettings
from harkd.services.processing_worker import ProcessingWorker
from harkd.storage.models import StorageRecording


@pytest.fixture(scope="function")
def settings(tmp_path_factory):
    """Create test settings with temp storage.

    Function-scoped to ensure each test gets a fresh storage directory.
    """
    # Create a unique temp directory for each test
    tmp_dir = tmp_path_factory.mktemp("recordings")
    return HarkdSettings(storage=StorageSettings(base_path=tmp_dir))


@pytest.fixture
def client(settings):
    """Create test client."""
    # Clear service caches to ensure fresh state per test
    from harkd.api import deps
    from harkd.config import get_settings

    deps._recording_services.clear()
    deps._voice_profile_services.clear()
    deps._processing_workers.clear()

    # Reset global recording state
    from harkd.state.recording_state import get_recording_state

    get_recording_state.cache_clear()

    # Ensure storage directory exists and is empty
    base_path = settings.storage.base_path
    base_path.mkdir(parents=True, exist_ok=True)

    app = create_app(settings)

    # Override the get_settings dependency to use our test settings
    app.dependency_overrides[get_settings] = lambda: settings

    return TestClient(app)


@pytest.fixture
def mock_recorder():
    """Mock AudioRecorder."""
    with patch("harkd.services.recording_service.AudioRecorder") as mock:
        instance = MagicMock()
        mock.return_value = instance
        yield instance


def create_recording_in_storage(settings, recording):
    """Helper to create a recording directly in storage.

    This creates the recording by writing the JSON file directly to the filesystem,
    ensuring it's visible to the API without needing async context.

    Args:
        settings: Test settings
        recording: StorageRecording to create
    """
    import json

    # Storage uses {base_path}/recordings/{recording-id}/ structure
    recordings_dir = settings.storage.base_path / "recordings"
    recording_dir = recordings_dir / recording.id
    recording_dir.mkdir(parents=True, exist_ok=True)

    # Write metadata.json
    metadata_path = recording_dir / "metadata.json"
    metadata = {
        "id": recording.id,
        "status": recording.status,
        "created_at": recording.created_at.isoformat(),
        "title": recording.title,
        "duration": recording.duration,
        "mic_enabled": recording.mic_enabled,
        "speaker_enabled": recording.speaker_enabled,
        "mic_level": recording.mic_level,
        "speaker_level": recording.speaker_level,
        "processing_stage": recording.processing_stage,
        "processing_progress": recording.processing_progress,
        "retry_count": recording.retry_count,
        "max_retries": recording.max_retries,
        "last_error": recording.last_error,
        "last_error_at": (recording.last_error_at.isoformat() if recording.last_error_at else None),
        "error_history": recording.error_history,
        "model": recording.model,
        "language": recording.language,
        "language_confidence": recording.language_confidence,
        "diarized": recording.diarized,
        "speakers": recording.speakers or [],
        "speaker_embeddings": recording.speaker_embeddings,
        "speaker_profiles": recording.speaker_profiles,
        "segments": recording.segments or [],
        "transcript": recording.transcript,
        "tags": recording.tags or [],
        "executive_summary": recording.executive_summary or [],
        "meeting_notes": recording.meeting_notes or [],
        "tasks": recording.tasks or [],
        "decisions": recording.decisions or [],
        "settings": recording.settings,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


@pytest.fixture
def mock_transcriber():
    """Mock transcription subprocess on the ProcessingWorker."""
    mock_result = {
        "text": "Hello world",
        "language": "en",
        "language_probability": 0.95,
        "duration": 2.0,
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world",
                "words": [
                    {"start": 0.0, "end": 1.0, "word": "Hello"},
                    {"start": 1.0, "end": 2.0, "word": "world"},
                ],
            }
        ],
    }

    async def mock_run_subprocess(self, *args, **kwargs):
        return mock_result

    with patch.object(ProcessingWorker, "_run_transcription", mock_run_subprocess):
        yield mock_result


class TestStartRecording:
    """Tests for POST /api/v1/recordings."""

    def test_start_recording_success(self, client, mock_recorder):
        """Test starting a recording."""
        response = client.post(
            "/api/v1/recordings",
            json={"title": "Test Recording"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "recording"
        assert data["title"] == "Test Recording"
        assert data["duration"] == 0.0
        assert "id" in data

    def test_start_recording_default_title(self, client, mock_recorder):
        """Test starting recording without title."""
        response = client.post("/api/v1/recordings", json={})

        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Untitled Recording"

    def test_start_recording_with_overrides(self, client, mock_recorder):
        """Test starting recording with per-recording overrides."""
        response = client.post(
            "/api/v1/recordings",
            json={
                "title": "Custom",
                "settings": {
                    "mic_enabled": True,
                    "speaker_enabled": False,
                    "language": "en",
                },
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["settings"]["mic_enabled"] is True
        assert data["settings"]["speaker_enabled"] is False
        assert data["settings"]["language"] == "en"
        # model comes from daemon defaults
        assert data["settings"]["model"] == "large-v3"

    def test_start_recording_without_settings_uses_defaults(self, client, mock_recorder):
        """Test starting recording without settings uses daemon defaults."""
        response = client.post(
            "/api/v1/recordings",
            json={"title": "Default Settings"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["settings"]["model"] == "large-v3"
        assert data["settings"]["mic_enabled"] is True
        assert data["settings"]["speaker_enabled"] is True
        assert data["settings"]["language"] == "auto"
        assert data["settings"]["diarization"] is True

    def test_start_recording_rejects_model_override(self, client, mock_recorder):
        """Test that model cannot be overridden per-recording (rejected by schema)."""
        response = client.post(
            "/api/v1/recordings",
            json={
                "title": "Override attempt",
                "settings": {
                    "model": "tiny",
                },
            },
        )

        # model is not in RecordingOverrides, so it should be ignored by pydantic
        # (extra fields are forbidden by default — but BaseModel ignores them)
        # The recording should use daemon default model
        assert response.status_code == 201
        data = response.json()
        assert data["settings"]["model"] == "large-v3"

    def test_start_recording_already_recording(self, client, mock_recorder):
        """Test error when starting while already recording."""
        # Start first recording
        response1 = client.post("/api/v1/recordings", json={})
        assert response1.status_code == 201

        # Try to start second
        response2 = client.post("/api/v1/recordings", json={})
        assert response2.status_code == 409
        error = response2.json()
        assert error["detail"]["error"]["code"] == "RECORDING_IN_PROGRESS"


class TestGetRecording:
    """Tests for GET /api/v1/recordings/{id}."""

    def test_get_recording(self, client, mock_recorder, settings):
        """Test getting a recording."""
        # Start recording
        response = client.post("/api/v1/recordings", json={"title": "Test"})
        recording_id = response.json()["id"]

        # Get recording
        response = client.get(f"/api/v1/recordings/{recording_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == recording_id
        assert data["title"] == "Test"
        assert data["status"] == "recording"

    def test_get_recording_not_found(self, client):
        """Test getting nonexistent recording."""
        response = client.get("/api/v1/recordings/nonexistent-id")

        assert response.status_code == 404
        error = response.json()
        assert error["detail"]["error"]["code"] == "RECORDING_NOT_FOUND"


class TestGetActiveRecording:
    """Tests for GET /api/v1/recordings/active."""

    def test_get_active_recording_success(self, client, mock_recorder):
        """Test getting the active recording."""
        # Start recording
        response = client.post("/api/v1/recordings", json={"title": "Test"})
        recording_id = response.json()["id"]

        # Get active recording
        response = client.get("/api/v1/recordings/active")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == recording_id
        assert data["title"] == "Test"
        assert data["status"] == "recording"

    def test_get_active_recording_no_active(self, client):
        """Test getting active recording when none exists."""
        response = client.get("/api/v1/recordings/active")

        assert response.status_code == 409
        error = response.json()
        assert error["detail"]["error"]["code"] == "NO_ACTIVE_RECORDING"

    def test_get_active_recording_real_time_duration(self, client, mock_recorder):
        """Test that duration updates in real-time for active recordings."""
        # Start recording
        start_response = client.post("/api/v1/recordings", json={"title": "Test"})
        assert start_response.status_code == 201
        recording_id = start_response.json()["id"]

        # Get active recording immediately
        response1 = client.get("/api/v1/recordings/active")
        assert response1.status_code == 200
        data1 = response1.json()
        duration1 = data1["duration"]
        assert data1["id"] == recording_id
        assert data1["status"] == "recording"
        assert duration1 >= 0

        # Wait a bit
        import time

        time.sleep(0.15)

        # Get active recording again
        response2 = client.get("/api/v1/recordings/active")
        assert response2.status_code == 200
        data2 = response2.json()
        duration2 = data2["duration"]
        assert data2["id"] == recording_id

        # Duration should have increased by approximately the wait time
        assert duration2 > duration1, f"Duration should increase: {duration1} -> {duration2}"
        assert duration2 >= 0.14, f"Duration should be at least 0.14s, got {duration2}s"

        # Verify getting by ID also shows updated duration
        response3 = client.get(f"/api/v1/recordings/{recording_id}")
        assert response3.status_code == 200
        duration3 = response3.json()["duration"]
        assert duration3 >= duration2, "GET /recordings/{id} should also show real-time duration"


class TestStopActiveRecording:
    """Tests for POST /api/v1/recordings/active/stop."""

    def test_stop_active_recording_success(self, client, mock_recorder, mock_transcriber, settings):
        """Test stopping the active recording."""
        # Start recording
        response = client.post("/api/v1/recordings", json={"title": "Test"})
        recording_id = response.json()["id"]

        # Create audio file (required for processing)
        audio_dir = settings.storage.base_path / "recordings" / recording_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / "audio.wav"
        audio_path.touch()

        # Stop active recording
        response = client.post("/api/v1/recordings/active/stop")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == recording_id
        assert data["status"] == "processing"
        assert data["duration"] > 0

    def test_stop_active_recording_no_active(self, client):
        """Test stopping when no recording is active."""
        response = client.post("/api/v1/recordings/active/stop")

        assert response.status_code == 409
        error = response.json()
        assert error["detail"]["error"]["code"] == "NO_ACTIVE_RECORDING"
        assert error["detail"]["error"]["message"] == "No active recording"

    def test_active_recording_lifecycle(self, client, mock_recorder, mock_transcriber, settings):
        """Test complete lifecycle: start -> check active -> stop via active endpoint."""
        # Initially no active recording
        response = client.get("/api/v1/recordings/active")
        assert response.status_code == 409, "Should be 409 when no recording active"

        # Start recording
        start_response = client.post("/api/v1/recordings", json={"title": "Lifecycle Test"})
        assert start_response.status_code == 201
        recording_id = start_response.json()["id"]

        # Now there should be an active recording
        active_response = client.get("/api/v1/recordings/active")
        assert active_response.status_code == 200
        active_data = active_response.json()
        assert active_data["id"] == recording_id
        assert active_data["status"] == "recording"
        assert active_data["title"] == "Lifecycle Test"

        # Create audio file for processing
        import time

        time.sleep(0.1)  # Let some time pass
        audio_dir = settings.storage.base_path / "recordings" / recording_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / "audio.wav"
        audio_path.touch()

        # Stop via active endpoint (no ID needed)
        stop_response = client.post("/api/v1/recordings/active/stop")
        assert stop_response.status_code == 200
        stop_data = stop_response.json()
        assert stop_data["id"] == recording_id
        assert stop_data["status"] == "processing"
        assert stop_data["duration"] > 0, "Should have recorded some duration"

        # After stopping, no active recording should exist
        response = client.get("/api/v1/recordings/active")
        assert response.status_code == 409, "Should be 409 after stopping"

        # Recording should still exist but in processing state
        get_response = client.get(f"/api/v1/recordings/{recording_id}")
        assert get_response.status_code == 200
        assert get_response.json()["status"] == "processing"


class TestListRecordings:
    """Tests for GET /api/v1/recordings."""

    def test_list_recordings_empty(self, client):
        """Test listing when no recordings."""
        response = client.get("/api/v1/recordings")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["recordings"] == []

    def test_list_recordings(self, client, settings):
        """Test listing recordings."""
        # Create some recordings with different statuses
        recordings_data = [
            ("test-1", "Complete Recording", RecordingStatus.COMPLETE.value, 10.0),
            ("test-2", "Processing Recording", RecordingStatus.PROCESSING.value, 5.0),
            ("test-3", "Another Complete", RecordingStatus.COMPLETE.value, 15.0),
        ]

        for rec_id, title, status, duration in recordings_data:
            recording = StorageRecording(
                id=rec_id,
                status=status,
                created_at=datetime.now(UTC),
                title=title,
                duration=duration,
                settings={},
            )
            create_recording_in_storage(settings, recording)

        # List all recordings
        response = client.get("/api/v1/recordings")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["recordings"]) == 3

        # Verify recordings contain expected fields
        for recording in data["recordings"]:
            assert "id" in recording
            assert "title" in recording
            assert "status" in recording
            assert "duration" in recording
            assert "created_at" in recording

        # Verify specific recordings are present
        titles = {rec["title"] for rec in data["recordings"]}
        assert "Complete Recording" in titles
        assert "Processing Recording" in titles
        assert "Another Complete" in titles

    def test_list_recordings_pagination(self, client, settings):
        """Test pagination works correctly and returns different results."""
        # Create 5 recordings with predictable IDs
        import time

        for i in range(5):
            recording = StorageRecording(
                id=f"test-{i:03d}",  # Padded for consistent ordering
                status=RecordingStatus.COMPLETE.value,
                created_at=datetime.now(UTC),
                title=f"Recording {i}",
                duration=10.0 + i,  # Different durations for distinction
                settings={},
            )
            create_recording_in_storage(settings, recording)
            time.sleep(0.001)  # Ensure different creation times

        # Get first page
        response1 = client.get("/api/v1/recordings?limit=2&offset=0")
        assert response1.status_code == 200
        data1 = response1.json()
        assert len(data1["recordings"]) == 2
        assert data1["limit"] == 2
        assert data1["offset"] == 0

        # Get second page
        response2 = client.get("/api/v1/recordings?limit=2&offset=2")
        assert response2.status_code == 200
        data2 = response2.json()
        assert len(data2["recordings"]) == 2
        assert data2["limit"] == 2
        assert data2["offset"] == 2

        # Get third page (remaining item)
        response3 = client.get("/api/v1/recordings?limit=2&offset=4")
        assert response3.status_code == 200
        data3 = response3.json()
        assert len(data3["recordings"]) == 1

        # Verify pages don't overlap (no duplicate IDs across pages)
        ids_page1 = {rec["id"] for rec in data1["recordings"]}
        ids_page2 = {rec["id"] for rec in data2["recordings"]}
        ids_page3 = {rec["id"] for rec in data3["recordings"]}

        assert len(ids_page1 & ids_page2) == 0, "Page 1 and 2 should not overlap"
        assert len(ids_page1 & ids_page3) == 0, "Page 1 and 3 should not overlap"
        assert len(ids_page2 & ids_page3) == 0, "Page 2 and 3 should not overlap"

        # Verify all recordings are accounted for
        all_ids = ids_page1 | ids_page2 | ids_page3
        assert len(all_ids) == 5, "All 5 recordings should be present across pages"

    def test_list_recordings_filter_by_status(self, client, settings):
        """Test filtering by status returns only matching recordings."""
        # Create recordings with different statuses
        recordings_data = [
            ("complete-1", "Complete 1", RecordingStatus.COMPLETE.value),
            ("complete-2", "Complete 2", RecordingStatus.COMPLETE.value),
            ("processing-1", "Processing 1", RecordingStatus.PROCESSING.value),
            ("recording-1", "Recording 1", RecordingStatus.RECORDING.value),
        ]

        for rec_id, title, status in recordings_data:
            create_recording_in_storage(
                settings,
                StorageRecording(
                    id=rec_id,
                    status=status,
                    created_at=datetime.now(UTC),
                    title=title,
                    duration=10.0,
                    settings={},
                ),
            )

        # Filter by complete status
        response = client.get("/api/v1/recordings?status=complete")
        assert response.status_code == 200
        data = response.json()
        assert len(data["recordings"]) == 2, "Should have exactly 2 complete recordings"
        assert all(r["status"] == "complete" for r in data["recordings"])
        titles = {r["title"] for r in data["recordings"]}
        assert titles == {"Complete 1", "Complete 2"}

        # Filter by processing status
        response = client.get("/api/v1/recordings?status=processing")
        assert response.status_code == 200
        data = response.json()
        assert len(data["recordings"]) == 1, "Should have exactly 1 processing recording"
        assert data["recordings"][0]["status"] == "processing"
        assert data["recordings"][0]["title"] == "Processing 1"

        # Filter by recording status
        response = client.get("/api/v1/recordings?status=recording")
        assert response.status_code == 200
        data = response.json()
        assert len(data["recordings"]) == 1, "Should have exactly 1 active recording"
        assert data["recordings"][0]["status"] == "recording"
        assert data["recordings"][0]["title"] == "Recording 1"

        # No filter - should get all recordings
        response = client.get("/api/v1/recordings")
        assert response.status_code == 200
        data = response.json()
        assert len(data["recordings"]) == 4, "Should have all 4 recordings without filter"

    def test_list_recordings_sort_asc(self, client, settings):
        """Test sort=asc returns oldest first, default returns newest first."""
        base = datetime(2026, 1, 10, 12, 0, 0)
        for i in range(3):
            create_recording_in_storage(
                settings,
                StorageRecording(
                    id=f"rec-{i}",
                    status=RecordingStatus.COMPLETE.value,
                    created_at=base + timedelta(hours=i),
                    title=f"Recording {i}",
                    duration=10.0,
                    settings={},
                ),
            )

        # Default (desc) — newest first
        response = client.get("/api/v1/recordings")
        assert response.status_code == 200
        ids_desc = [r["id"] for r in response.json()["recordings"]]
        assert ids_desc == ["rec-2", "rec-1", "rec-0"]

        # Explicit asc — oldest first
        response = client.get("/api/v1/recordings?sort=asc")
        assert response.status_code == 200
        ids_asc = [r["id"] for r in response.json()["recordings"]]
        assert ids_asc == ["rec-0", "rec-1", "rec-2"]

    def test_list_recordings_filter_created_after(self, client, settings):
        """Test created_after returns only recordings at or after the cutoff."""
        base = datetime(2026, 1, 10, 12, 0, 0, tzinfo=UTC)
        for i in range(3):
            create_recording_in_storage(
                settings,
                StorageRecording(
                    id=f"rec-{i}",
                    status=RecordingStatus.COMPLETE.value,
                    created_at=base + timedelta(hours=i),
                    title=f"Recording {i}",
                    duration=10.0,
                    settings={},
                ),
            )

        # Use Z suffix (not +00:00) to avoid URL encoding issues with +
        cutoff = (base + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        response = client.get(f"/api/v1/recordings?created_after={cutoff}")
        assert response.status_code == 200
        ids = {r["id"] for r in response.json()["recordings"]}
        assert ids == {"rec-1", "rec-2"}

    def test_list_recordings_filter_created_before(self, client, settings):
        """Test created_before returns only recordings at or before the cutoff."""
        base = datetime(2026, 1, 10, 12, 0, 0, tzinfo=UTC)
        for i in range(3):
            create_recording_in_storage(
                settings,
                StorageRecording(
                    id=f"rec-{i}",
                    status=RecordingStatus.COMPLETE.value,
                    created_at=base + timedelta(hours=i),
                    title=f"Recording {i}",
                    duration=10.0,
                    settings={},
                ),
            )

        cutoff = (base + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        response = client.get(f"/api/v1/recordings?created_before={cutoff}")
        assert response.status_code == 200
        ids = {r["id"] for r in response.json()["recordings"]}
        assert ids == {"rec-0", "rec-1"}

    def test_list_recordings_date_range(self, client, settings):
        """Test combining created_after and created_before filters."""
        base = datetime(2026, 1, 10, 12, 0, 0, tzinfo=UTC)
        for i in range(5):
            create_recording_in_storage(
                settings,
                StorageRecording(
                    id=f"rec-{i}",
                    status=RecordingStatus.COMPLETE.value,
                    created_at=base + timedelta(hours=i),
                    title=f"Recording {i}",
                    duration=10.0,
                    settings={},
                ),
            )

        after = (base + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        before = (base + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        response = client.get(f"/api/v1/recordings?created_after={after}&created_before={before}")
        assert response.status_code == 200
        ids = {r["id"] for r in response.json()["recordings"]}
        assert ids == {"rec-1", "rec-2", "rec-3"}

    def test_list_recordings_combined_filters(self, client, settings):
        """Test combining status + search + date range + sort."""
        base = datetime(2026, 1, 10, 12, 0, 0)
        recordings = [
            ("rec-0", "Team standup", "complete", base),
            ("rec-1", "Team retro", "complete", base + timedelta(hours=1)),
            ("rec-2", "Team standup", "processing", base + timedelta(hours=2)),
            ("rec-3", "Design review", "complete", base + timedelta(hours=3)),
            ("rec-4", "Team standup", "complete", base + timedelta(hours=4)),
        ]
        for rec_id, title, status, created_at in recordings:
            create_recording_in_storage(
                settings,
                StorageRecording(
                    id=rec_id,
                    status=status,
                    created_at=created_at,
                    title=title,
                    duration=10.0,
                    settings={},
                ),
            )

        # Complete + "standup" + date range + asc
        after = (base + timedelta(hours=1)).isoformat()
        before = (base + timedelta(hours=4)).isoformat()
        response = client.get(
            f"/api/v1/recordings?status=complete&search=standup"
            f"&created_after={after}&created_before={before}&sort=asc"
        )
        assert response.status_code == 200
        data = response.json()
        ids = [r["id"] for r in data["recordings"]]
        # rec-0: before range, rec-2: processing, rec-3: no match
        assert ids == ["rec-4"]


class TestUpdateRecording:
    """Tests for PATCH /api/v1/recordings/{id}."""

    def test_update_recording_title(self, client, settings):
        """Test updating recording title persists the change."""
        recording = StorageRecording(
            id="test-123",
            status=RecordingStatus.COMPLETE.value,
            created_at=datetime.now(UTC),
            title="Old Title",
            duration=10.0,
            settings={},
        )
        create_recording_in_storage(settings, recording)

        # Update title
        response = client.patch(
            "/api/v1/recordings/test-123",
            json={"title": "New Title"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "New Title"
        assert data["id"] == "test-123"
        assert data["status"] == "complete"

        # Verify the update persisted by fetching the recording again
        response = client.get("/api/v1/recordings/test-123")
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "New Title", "Title change should persist"

        # Verify it appears in list with new title
        response = client.get("/api/v1/recordings")
        assert response.status_code == 200
        data = response.json()
        recording_titles = {r["title"] for r in data["recordings"]}
        assert "New Title" in recording_titles
        assert "Old Title" not in recording_titles

    def test_update_recording_speakers(self, client, settings):
        """Test updating speaker names updates throughout segments and words."""
        recording = StorageRecording(
            id="test-123",
            status=RecordingStatus.COMPLETE.value,
            created_at=datetime.now(UTC),
            title="Test",
            duration=10.0,
            segments=[
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello there",
                    "speaker": "SPEAKER_01",
                    "words": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "word": "Hello",
                            "speaker": "SPEAKER_01",
                        },
                        {
                            "start": 1.0,
                            "end": 2.5,
                            "word": "there",
                            "speaker": "SPEAKER_01",
                        },
                    ],
                },
                {
                    "start": 2.5,
                    "end": 5.0,
                    "text": "How are you",
                    "speaker": "SPEAKER_02",
                    "words": [
                        {
                            "start": 2.5,
                            "end": 3.0,
                            "word": "How",
                            "speaker": "SPEAKER_02",
                        },
                        {
                            "start": 3.0,
                            "end": 3.5,
                            "word": "are",
                            "speaker": "SPEAKER_02",
                        },
                        {
                            "start": 3.5,
                            "end": 5.0,
                            "word": "you",
                            "speaker": "SPEAKER_02",
                        },
                    ],
                },
            ],
            speakers=["SPEAKER_01", "SPEAKER_02"],
            settings={},
        )
        create_recording_in_storage(settings, recording)

        # Update speakers
        response = client.patch(
            "/api/v1/recordings/test-123",
            json={"speakers": {"SPEAKER_01": "Alice", "SPEAKER_02": "Bob"}},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify speaker list updated
        assert set(data["speakers"]) == {"Alice", "Bob"}

        # Verify segment speakers updated
        assert data["segments"][0]["speaker"] == "Alice"
        assert data["segments"][1]["speaker"] == "Bob"

        # Verify word-level speakers updated
        assert all(w["speaker"] == "Alice" for w in data["segments"][0]["words"])
        assert all(w["speaker"] == "Bob" for w in data["segments"][1]["words"])

        # Verify update persisted
        response = client.get("/api/v1/recordings/test-123")
        assert response.status_code == 200
        data = response.json()
        assert set(data["speakers"]) == {"Alice", "Bob"}
        assert "SPEAKER_01" not in str(data["segments"]), "Old speaker IDs should not remain"
        assert "SPEAKER_02" not in str(data["segments"]), "Old speaker IDs should not remain"

    def test_update_recording_speakers_not_complete(self, client, settings):
        """Test error when updating speakers on non-complete recording."""
        recording = StorageRecording(
            id="test-123",
            status=RecordingStatus.PROCESSING.value,
            created_at=datetime.now(UTC),
            title="Test",
            duration=10.0,
            settings={},
        )
        create_recording_in_storage(settings, recording)

        # Try to update speakers — should be rejected
        response = client.patch(
            "/api/v1/recordings/test-123",
            json={"speakers": {"SPEAKER_01": "Alice"}},
        )

        assert response.status_code == 409
        error = response.json()
        assert error["detail"]["error"]["code"] == "INVALID_STATE"

    def test_update_title_during_processing(self, client, settings):
        """Test that title can be updated on a processing recording."""
        recording = StorageRecording(
            id="test-123",
            status=RecordingStatus.PROCESSING.value,
            created_at=datetime.now(UTC),
            title="Untitled Recording",
            duration=10.0,
            settings={},
        )
        create_recording_in_storage(settings, recording)

        response = client.patch(
            "/api/v1/recordings/test-123",
            json={"title": "Weekly Standup"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Weekly Standup"

    def test_update_title_during_recording(self, client, settings):
        """Test that title can be updated on a recording-state recording."""
        recording = StorageRecording(
            id="test-123",
            status=RecordingStatus.RECORDING.value,
            created_at=datetime.now(UTC),
            title="Untitled Recording",
            duration=0.0,
            settings={},
        )
        create_recording_in_storage(settings, recording)

        response = client.patch(
            "/api/v1/recordings/test-123",
            json={"title": "My Meeting"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "My Meeting"

    def test_update_tasks_via_patch(self, client, settings):
        """Test updating tasks via PATCH persists and is retrievable."""
        recording = StorageRecording(
            id="test-tasks",
            status=RecordingStatus.COMPLETE.value,
            created_at=datetime.now(UTC),
            title="Test",
            duration=10.0,
            tasks=[],
            settings={},
        )
        create_recording_in_storage(settings, recording)

        tasks = [{"task": "Review PR", "done": False}, {"task": "Deploy", "done": True}]
        response = client.patch(
            "/api/v1/recordings/test-tasks",
            json={"tasks": tasks},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["tasks"]) == 2
        assert data["tasks"][0]["task"] == "Review PR"
        assert data["tasks"][1]["done"] is True

        # Verify persisted via GET
        response = client.get("/api/v1/recordings/test-tasks")
        assert response.status_code == 200
        assert len(response.json()["tasks"]) == 2

        # Verify list endpoint still works
        response = client.get("/api/v1/recordings")
        assert response.status_code == 200
        assert response.json()["total"] == 1

    def test_update_decisions_via_patch(self, client, settings):
        """Test updating decisions via PATCH persists and is retrievable."""
        recording = StorageRecording(
            id="test-decisions",
            status=RecordingStatus.COMPLETE.value,
            created_at=datetime.now(UTC),
            title="Test",
            duration=10.0,
            decisions=[],
            settings={},
        )
        create_recording_in_storage(settings, recording)

        decisions = ["Use React", "Deploy on Friday"]
        response = client.patch(
            "/api/v1/recordings/test-decisions",
            json={"decisions": decisions},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["decisions"] == ["Use React", "Deploy on Friday"]

        # Verify persisted via GET
        response = client.get("/api/v1/recordings/test-decisions")
        assert response.status_code == 200
        assert response.json()["decisions"] == ["Use React", "Deploy on Friday"]

    def test_tags_in_list_response(self, client, settings):
        """Test that tags appear correctly in list response items."""
        recording = StorageRecording(
            id="test-tags",
            status=RecordingStatus.COMPLETE.value,
            created_at=datetime.now(UTC),
            title="Tagged Recording",
            duration=10.0,
            tags=["standup", "weekly"],
            settings={},
        )
        create_recording_in_storage(settings, recording)

        response = client.get("/api/v1/recordings")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["recordings"][0]["tags"] == ["standup", "weekly"]


class TestDeleteRecording:
    """Tests for DELETE /api/v1/recordings/{id}."""

    def test_delete_complete_recording(self, client, settings):
        """Test deleting a complete recording removes it completely."""
        # Create multiple recordings
        for i in range(3):
            recording = StorageRecording(
                id=f"test-{i}",
                status=RecordingStatus.COMPLETE.value,
                created_at=datetime.now(UTC),
                title=f"Recording {i}",
                duration=10.0,
                settings={},
            )
            create_recording_in_storage(settings, recording)

        # Verify all exist
        response = client.get("/api/v1/recordings")
        assert response.status_code == 200
        assert response.json()["total"] == 3

        # Delete one recording
        response = client.delete("/api/v1/recordings/test-1")
        assert response.status_code == 204
        assert response.text == "", "Delete should return empty body"

        # Verify it's gone
        response = client.get("/api/v1/recordings/test-1")
        assert response.status_code == 404

        # Verify it's not in the list
        response = client.get("/api/v1/recordings")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2, "Should have 2 recordings after deleting 1"
        recording_ids = {r["id"] for r in data["recordings"]}
        assert "test-1" not in recording_ids
        assert "test-0" in recording_ids
        assert "test-2" in recording_ids

        # Verify physical directory is removed
        recording_dir = settings.storage.base_path / "recordings" / "test-1"
        assert not recording_dir.exists(), "Recording directory should be deleted"

    def test_delete_recording_not_found(self, client):
        """Test deleting nonexistent recording."""
        response = client.delete("/api/v1/recordings/nonexistent-id")

        assert response.status_code == 404
        error = response.json()
        assert error["detail"]["error"]["code"] == "RECORDING_NOT_FOUND"

    def test_delete_active_recording(self, client, mock_recorder):
        """Test deleting an active (recording-state) recording stops and removes it."""
        # Start a recording
        response = client.post("/api/v1/recordings", json={"title": "Active Test"})
        assert response.status_code == 201
        recording_id = response.json()["id"]

        # Verify it's active
        response = client.get("/api/v1/recordings/active")
        assert response.status_code == 200
        assert response.json()["id"] == recording_id

        # Delete the active recording
        response = client.delete(f"/api/v1/recordings/{recording_id}")
        assert response.status_code == 204

        # Verify recording is gone
        response = client.get(f"/api/v1/recordings/{recording_id}")
        assert response.status_code == 404

    def test_delete_active_recording_clears_active_state(self, client, mock_recorder):
        """Test that deleting an active recording clears the active recording state."""
        # Start a recording
        response = client.post("/api/v1/recordings", json={"title": "Clear State Test"})
        assert response.status_code == 201
        recording_id = response.json()["id"]

        # Delete it
        response = client.delete(f"/api/v1/recordings/{recording_id}")
        assert response.status_code == 204

        # GET /active should now return 409 (no active recording)
        response = client.get("/api/v1/recordings/active")
        assert response.status_code == 409
        assert response.json()["detail"]["error"]["code"] == "NO_ACTIVE_RECORDING"

    def test_delete_processing_recording(self, client, settings):
        """Test deleting a processing recording removes it."""
        recording = StorageRecording(
            id="proc-001",
            status=RecordingStatus.PROCESSING.value,
            created_at=datetime.now(UTC),
            title="Processing Recording",
            duration=15.0,
            settings={},
        )
        create_recording_in_storage(settings, recording)

        # Verify it exists
        response = client.get("/api/v1/recordings/proc-001")
        assert response.status_code == 200
        assert response.json()["status"] == "processing"

        # Delete it
        response = client.delete("/api/v1/recordings/proc-001")
        assert response.status_code == 204

        # Verify it's gone
        response = client.get("/api/v1/recordings/proc-001")
        assert response.status_code == 404

        # Verify physical directory is removed
        recording_dir = settings.storage.base_path / "recordings" / "proc-001"
        assert not recording_dir.exists()

    def test_delete_allows_new_recording(self, client, mock_recorder):
        """Test that after deleting an active recording, a new one can be started."""
        # Start a recording
        response = client.post("/api/v1/recordings", json={"title": "First"})
        assert response.status_code == 201
        first_id = response.json()["id"]

        # Delete it
        response = client.delete(f"/api/v1/recordings/{first_id}")
        assert response.status_code == 204

        # Start a new recording — should succeed (no conflict)
        response = client.post("/api/v1/recordings", json={"title": "Second"})
        assert response.status_code == 201
        assert response.json()["title"] == "Second"


class TestRetryRecording:
    """Tests for POST /api/v1/recordings/{id}/retry."""

    def test_retry_recording_success(self, client, settings):
        """Test retrying a failed recording."""
        recording = StorageRecording(
            id="failed-001",
            status="error",
            created_at=datetime.now(UTC),
            title="Failed Recording",
            duration=10.0,
            settings={},
            retry_count=1,
            last_error="Transcription timed out after 1800s",
        )
        create_recording_in_storage(settings, recording)

        response = client.post("/api/v1/recordings/failed-001/retry")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["id"] == "failed-001"

    def test_retry_recording_not_found(self, client):
        """Test retrying a nonexistent recording."""
        response = client.post("/api/v1/recordings/nonexistent-id/retry")

        assert response.status_code == 404

    def test_retry_recording_wrong_status(self, client, settings):
        """Test retrying a processing recording returns 409."""
        recording = StorageRecording(
            id="processing-001",
            status="processing",
            created_at=datetime.now(UTC),
            title="Processing Recording",
            duration=10.0,
            settings={},
        )
        create_recording_in_storage(settings, recording)

        response = client.post("/api/v1/recordings/processing-001/retry")

        assert response.status_code == 409
        error = response.json()
        assert error["detail"]["error"]["code"] == "RETRY_NOT_ALLOWED"

    def test_retry_response_includes_retry_fields(self, client, settings):
        """Test that retry response includes retry_count and last_error."""
        recording = StorageRecording(
            id="err-001",
            status="error",
            created_at=datetime.now(UTC),
            title="Error Recording",
            duration=10.0,
            settings={},
            retry_count=2,
            last_error="OOM killed",
        )
        create_recording_in_storage(settings, recording)

        # Check the GET response includes retry fields
        response = client.get("/api/v1/recordings/err-001")
        assert response.status_code == 200
        data = response.json()
        assert data["retry_count"] == 2
        assert data["last_error"] == "OOM killed"


class TestSpeakerProfileIds:
    """Tests for PATCH /api/v1/recordings/{id} with speaker_profile_ids."""

    def test_update_with_speaker_profile_ids(self, client, settings):
        """Test linking speakers to existing voice profiles via PATCH."""
        # Create a voice profile first
        response = client.post(
            "/api/v1/voice-profiles",
            json={"name": "Alice"},
        )
        assert response.status_code == 201
        profile_id = response.json()["id"]

        # Create a completed recording with speaker embeddings
        recording = StorageRecording(
            id="test-profile-link",
            status=RecordingStatus.COMPLETE.value,
            created_at=datetime.now(UTC),
            title="Test",
            duration=10.0,
            segments=[
                {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00", "words": []},
            ],
            speakers=["SPEAKER_00"],
            speaker_embeddings={"SPEAKER_00": [0.1, 0.2, 0.3]},
            settings={},
        )
        create_recording_in_storage(settings, recording)

        # PATCH with speaker_profile_ids
        response = client.patch(
            "/api/v1/recordings/test-profile-link",
            json={
                "speakers": {"SPEAKER_00": "Alice"},
                "speaker_profile_ids": {"SPEAKER_00": profile_id},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["speaker_profiles"]["SPEAKER_00"] == profile_id

    def test_update_with_invalid_profile_id_returns_422(self, client, settings):
        """Test that an invalid profile ID returns 422."""
        recording = StorageRecording(
            id="test-bad-profile",
            status=RecordingStatus.COMPLETE.value,
            created_at=datetime.now(UTC),
            title="Test",
            duration=10.0,
            segments=[
                {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00", "words": []},
            ],
            speakers=["SPEAKER_00"],
            speaker_embeddings={"SPEAKER_00": [0.1, 0.2, 0.3]},
            settings={},
        )
        create_recording_in_storage(settings, recording)

        response = client.patch(
            "/api/v1/recordings/test-bad-profile",
            json={
                "speakers": {"SPEAKER_00": "Alice"},
                "speaker_profile_ids": {"SPEAKER_00": "nonexistent-profile"},
            },
        )

        assert response.status_code == 422
        error = response.json()
        assert error["detail"]["error"]["code"] == "VOICE_PROFILE_NOT_FOUND"
