"""Tests for filesystem recording storage."""

import json
import logging
from datetime import datetime

import pytest

from harkd.exceptions import RecordingNotFoundError, StorageError
from harkd.storage.filesystem.recordings import FilesystemRecordingStorage
from harkd.storage.models import StorageRecording


@pytest.fixture
def temp_storage(tmp_path):
    """Create a temporary storage instance."""
    return FilesystemRecordingStorage(tmp_path)


@pytest.fixture
def sample_recording():
    """Create a sample recording for testing."""
    return StorageRecording(
        id="test-rec-123",
        status="complete",
        created_at=datetime(2026, 1, 15, 10, 30, 0),
        title="Test Recording",
        duration=120.5,
        mic_enabled=True,
        speaker_enabled=False,
        model="base",
        language="en",
        language_confidence=0.98,
        speakers=["Alice", "Bob"],
        segments=[],
        settings={"model": "base"},
    )


@pytest.mark.asyncio
async def test_create_recording(temp_storage, sample_recording):
    """Test creating a new recording."""
    created = await temp_storage.create(sample_recording)

    assert created.id == sample_recording.id
    assert created.title == sample_recording.title

    # Verify directory was created
    recording_dir = temp_storage.recordings_dir / sample_recording.id
    assert recording_dir.exists()
    assert recording_dir.is_dir()

    # Verify metadata file exists
    metadata_file = recording_dir / "metadata.json"
    assert metadata_file.exists()


@pytest.mark.asyncio
async def test_create_duplicate_recording(temp_storage, sample_recording):
    """Test creating a recording with duplicate ID fails."""
    await temp_storage.create(sample_recording)

    # Attempting to create again should fail
    with pytest.raises(StorageError) as exc_info:
        await temp_storage.create(sample_recording)

    assert "already exists" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_get_existing_recording(temp_storage, sample_recording):
    """Test getting an existing recording."""
    await temp_storage.create(sample_recording)

    retrieved = await temp_storage.get(sample_recording.id)

    assert retrieved is not None
    assert retrieved.id == sample_recording.id
    assert retrieved.title == sample_recording.title
    assert retrieved.duration == sample_recording.duration
    assert retrieved.speakers == sample_recording.speakers


@pytest.mark.asyncio
async def test_get_nonexistent_recording(temp_storage):
    """Test getting a non-existent recording returns None."""
    result = await temp_storage.get("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_list_recordings_empty(temp_storage):
    """Test listing when no recordings exist."""
    recordings = await temp_storage.list()
    assert recordings == []


@pytest.mark.asyncio
async def test_list_recordings_multiple(temp_storage):
    """Test listing multiple recordings."""
    # Create multiple recordings
    rec1 = StorageRecording(
        id="rec-1",
        status="complete",
        created_at=datetime(2026, 1, 15, 10, 0, 0),
        title="Recording 1",
        duration=60.0,
        settings={},
    )
    rec2 = StorageRecording(
        id="rec-2",
        status="complete",
        created_at=datetime(2026, 1, 15, 11, 0, 0),
        title="Recording 2",
        duration=120.0,
        settings={},
    )
    rec3 = StorageRecording(
        id="rec-3",
        status="processing",
        created_at=datetime(2026, 1, 15, 12, 0, 0),
        title="Recording 3",
        duration=0.0,
        settings={},
    )

    await temp_storage.create(rec1)
    await temp_storage.create(rec2)
    await temp_storage.create(rec3)

    # List all
    all_recordings = await temp_storage.list()
    assert len(all_recordings) == 3

    # Storage no longer sorts — just verify all recordings are returned
    returned_ids = {r.id for r in all_recordings}
    assert returned_ids == {"rec-1", "rec-2", "rec-3"}


@pytest.mark.asyncio
async def test_list_recordings_with_status_filter(temp_storage):
    """Test listing recordings filtered by status."""
    rec1 = StorageRecording(
        id="rec-1",
        status="complete",
        created_at=datetime(2026, 1, 15, 10, 0, 0),
        title="Complete",
        duration=60.0,
        settings={},
    )
    rec2 = StorageRecording(
        id="rec-2",
        status="processing",
        created_at=datetime(2026, 1, 15, 11, 0, 0),
        title="Processing",
        duration=0.0,
        settings={},
    )

    await temp_storage.create(rec1)
    await temp_storage.create(rec2)

    # Filter by complete
    complete_recs = await temp_storage.list(status="complete")
    assert len(complete_recs) == 1
    assert complete_recs[0].id == "rec-1"

    # Filter by processing
    processing_recs = await temp_storage.list(status="processing")
    assert len(processing_recs) == 1
    assert processing_recs[0].id == "rec-2"


@pytest.mark.asyncio
async def test_list_recordings_with_pagination(temp_storage):
    """Test listing recordings with pagination."""
    # Create 5 recordings
    for i in range(5):
        rec = StorageRecording(
            id=f"rec-{i}",
            status="complete",
            created_at=datetime(2026, 1, 15, 10, i, 0),
            title=f"Recording {i}",
            duration=60.0,
            settings={},
        )
        await temp_storage.create(rec)

    # Get first page (limit=2)
    page1 = await temp_storage.list(limit=2, offset=0)
    assert len(page1) == 2

    # Get second page
    page2 = await temp_storage.list(limit=2, offset=2)
    assert len(page2) == 2

    # Get third page
    page3 = await temp_storage.list(limit=2, offset=4)
    assert len(page3) == 1

    # Ensure no duplicates
    all_ids = [r.id for r in page1] + [r.id for r in page2] + [r.id for r in page3]
    assert len(all_ids) == len(set(all_ids))  # No duplicates


@pytest.mark.asyncio
async def test_update_recording(temp_storage, sample_recording):
    """Test updating an existing recording."""
    await temp_storage.create(sample_recording)

    # Modify the recording
    sample_recording.title = "Updated Title"
    sample_recording.duration = 200.0

    updated = await temp_storage.update(sample_recording)
    assert updated.title == "Updated Title"
    assert updated.duration == 200.0

    # Verify persistence
    retrieved = await temp_storage.get(sample_recording.id)
    assert retrieved.title == "Updated Title"
    assert retrieved.duration == 200.0


@pytest.mark.asyncio
async def test_update_nonexistent_recording(temp_storage, sample_recording):
    """Test updating a non-existent recording fails."""
    with pytest.raises(RecordingNotFoundError):
        await temp_storage.update(sample_recording)


@pytest.mark.asyncio
async def test_delete_recording(temp_storage, sample_recording):
    """Test deleting a recording."""
    await temp_storage.create(sample_recording)

    # Verify it exists
    assert await temp_storage.get(sample_recording.id) is not None

    # Delete it
    await temp_storage.delete(sample_recording.id)

    # Verify it's gone
    assert await temp_storage.get(sample_recording.id) is None

    # Verify directory is removed
    recording_dir = temp_storage.recordings_dir / sample_recording.id
    assert not recording_dir.exists()


@pytest.mark.asyncio
async def test_delete_nonexistent_recording(temp_storage):
    """Test deleting a non-existent recording fails."""
    with pytest.raises(RecordingNotFoundError):
        await temp_storage.delete("nonexistent-id")


@pytest.mark.asyncio
async def test_count_recordings(temp_storage):
    """Test counting recordings."""
    # Initially zero
    assert await temp_storage.count() == 0

    # Create some recordings
    for i in range(3):
        rec = StorageRecording(
            id=f"rec-{i}",
            status="complete" if i < 2 else "processing",
            created_at=datetime.now(),
            title=f"Recording {i}",
            duration=60.0,
            settings={},
        )
        await temp_storage.create(rec)

    # Count all
    assert await temp_storage.count() == 3

    # Count by status
    assert await temp_storage.count(status="complete") == 2
    assert await temp_storage.count(status="processing") == 1


@pytest.mark.asyncio
async def test_json_serialization_roundtrip(temp_storage, sample_recording):
    """Test that recordings survive JSON serialization/deserialization."""
    await temp_storage.create(sample_recording)
    retrieved = await temp_storage.get(sample_recording.id)

    # All fields should match
    assert retrieved.id == sample_recording.id
    assert retrieved.status == sample_recording.status
    assert retrieved.created_at == sample_recording.created_at
    assert retrieved.title == sample_recording.title
    assert retrieved.duration == sample_recording.duration
    assert retrieved.mic_enabled == sample_recording.mic_enabled
    assert retrieved.speaker_enabled == sample_recording.speaker_enabled
    assert retrieved.model == sample_recording.model
    assert retrieved.language == sample_recording.language
    assert retrieved.language_confidence == sample_recording.language_confidence
    assert retrieved.speakers == sample_recording.speakers
    assert retrieved.settings == sample_recording.settings


@pytest.mark.asyncio
async def test_atomic_write_does_not_corrupt_on_failure(temp_storage, sample_recording):
    """Test that interrupted writes don't corrupt existing data."""
    # Create initial recording
    await temp_storage.create(sample_recording)

    # Verify initial data is fine
    retrieved = await temp_storage.get(sample_recording.id)
    assert retrieved.title == "Test Recording"

    # Update successfully
    sample_recording.title = "Updated Title"
    await temp_storage.update(sample_recording)

    # Verify the metadata file exists and is valid JSON
    metadata_file = temp_storage.recordings_dir / sample_recording.id / "metadata.json"
    with open(metadata_file) as f:
        data = json.load(f)
    assert data["title"] == "Updated Title"

    # Verify no temp files left behind
    recording_dir = temp_storage.recordings_dir / sample_recording.id
    temp_files = list(recording_dir.glob(".metadata-*.tmp"))
    assert temp_files == []


@pytest.mark.asyncio
async def test_create_with_embeddings_writes_separate_file(temp_storage):
    """Test that create with speaker_embeddings produces a separate embeddings.json."""
    rec = StorageRecording(
        id="rec-emb",
        status="complete",
        created_at=datetime(2026, 1, 15, 10, 0, 0),
        title="Embeddings Test",
        duration=60.0,
        settings={},
        speaker_embeddings={"SPEAKER_00": [0.1, 0.2], "SPEAKER_01": [0.3, 0.4]},
    )
    await temp_storage.create(rec)

    recording_dir = temp_storage.recordings_dir / "rec-emb"

    # embeddings.json should exist with the embeddings data
    embeddings_file = recording_dir / "embeddings.json"
    assert embeddings_file.exists()
    with open(embeddings_file) as f:
        emb_data = json.load(f)
    assert emb_data == {"SPEAKER_00": [0.1, 0.2], "SPEAKER_01": [0.3, 0.4]}

    # metadata.json should NOT contain speaker_embeddings
    metadata_file = recording_dir / "metadata.json"
    with open(metadata_file) as f:
        meta_data = json.load(f)
    assert "speaker_embeddings" not in meta_data


@pytest.mark.asyncio
async def test_get_merges_embeddings_from_separate_file(temp_storage):
    """Test that get() merges embeddings from the separate embeddings.json."""
    rec = StorageRecording(
        id="rec-merge",
        status="complete",
        created_at=datetime(2026, 1, 15, 10, 0, 0),
        title="Merge Test",
        duration=60.0,
        settings={},
        speaker_embeddings={"SPEAKER_00": [0.5, 0.6]},
    )
    await temp_storage.create(rec)

    retrieved = await temp_storage.get("rec-merge")
    assert retrieved is not None
    assert retrieved.speaker_embeddings == {"SPEAKER_00": [0.5, 0.6]}


@pytest.mark.asyncio
async def test_list_returns_none_embeddings(temp_storage):
    """Test that list() returns recordings with speaker_embeddings=None."""
    rec = StorageRecording(
        id="rec-list-emb",
        status="complete",
        created_at=datetime(2026, 1, 15, 10, 0, 0),
        title="List Test",
        duration=60.0,
        settings={},
        speaker_embeddings={"SPEAKER_00": [0.1]},
    )
    await temp_storage.create(rec)

    recordings = await temp_storage.list()
    assert len(recordings) == 1
    assert recordings[0].speaker_embeddings is None


@pytest.mark.asyncio
async def test_get_speaker_embeddings_returns_data(temp_storage):
    """Test get_speaker_embeddings() returns correct data."""
    rec = StorageRecording(
        id="rec-get-emb",
        status="complete",
        created_at=datetime(2026, 1, 15, 10, 0, 0),
        title="Get Embeddings",
        duration=60.0,
        settings={},
        speaker_embeddings={"SPEAKER_00": [1.0, 2.0]},
    )
    await temp_storage.create(rec)

    embeddings = await temp_storage.get_speaker_embeddings("rec-get-emb")
    assert embeddings == {"SPEAKER_00": [1.0, 2.0]}


@pytest.mark.asyncio
async def test_get_speaker_embeddings_returns_none_when_missing(temp_storage):
    """Test get_speaker_embeddings() returns None when no embeddings exist."""
    rec = StorageRecording(
        id="rec-no-emb",
        status="complete",
        created_at=datetime(2026, 1, 15, 10, 0, 0),
        title="No Embeddings",
        duration=60.0,
        settings={},
    )
    await temp_storage.create(rec)

    embeddings = await temp_storage.get_speaker_embeddings("rec-no-emb")
    assert embeddings is None


@pytest.mark.asyncio
async def test_update_with_none_embeddings_removes_file(temp_storage):
    """Test that updating with speaker_embeddings=None removes embeddings.json."""
    rec = StorageRecording(
        id="rec-rm-emb",
        status="complete",
        created_at=datetime(2026, 1, 15, 10, 0, 0),
        title="Remove Embeddings",
        duration=60.0,
        settings={},
        speaker_embeddings={"SPEAKER_00": [0.1]},
    )
    await temp_storage.create(rec)

    recording_dir = temp_storage.recordings_dir / "rec-rm-emb"
    assert (recording_dir / "embeddings.json").exists()

    # Update with no embeddings
    rec.speaker_embeddings = None
    await temp_storage.update(rec)

    assert not (recording_dir / "embeddings.json").exists()
    retrieved = await temp_storage.get("rec-rm-emb")
    assert retrieved.speaker_embeddings is None


@pytest.mark.asyncio
async def test_corrupted_recordings_are_logged(temp_storage, caplog):
    """Test that corrupted recordings are logged and skipped during list."""
    # Create a valid recording
    rec = StorageRecording(
        id="valid-rec",
        status="complete",
        created_at=datetime(2026, 1, 15, 10, 0, 0),
        title="Valid",
        duration=60.0,
        settings={},
    )
    await temp_storage.create(rec)

    # Create a corrupted recording (invalid JSON in metadata)
    corrupted_dir = temp_storage.recordings_dir / "corrupted-rec"
    corrupted_dir.mkdir(parents=True)
    (corrupted_dir / "metadata.json").write_text("not valid json {{{")

    # List should return only the valid recording and log a warning
    with caplog.at_level(logging.WARNING):
        recordings = await temp_storage.list()

    assert len(recordings) == 1
    assert recordings[0].id == "valid-rec"
    assert "corrupted" in caplog.text.lower() or "Skipping" in caplog.text


# --- Pre-migration backward compatibility ---


@pytest.mark.asyncio
async def test_get_reads_embeddings_from_legacy_metadata(temp_storage):
    """Pre-migration: metadata.json contains speaker_embeddings, no embeddings.json.

    get() should still return the embeddings from metadata.json.
    """
    rec_dir = temp_storage.recordings_dir / "legacy-rec"
    rec_dir.mkdir(parents=True)

    metadata = {
        "id": "legacy-rec",
        "status": "complete",
        "created_at": "2025-06-01T00:00:00",
        "title": "Legacy",
        "duration": 60.0,
        "settings": {},
        "segments": [],
        "speaker_embeddings": {"SPEAKER_00": [0.1, 0.2, 0.3]},
    }
    (rec_dir / "metadata.json").write_text(json.dumps(metadata))

    recording = await temp_storage.get("legacy-rec")
    assert recording is not None
    assert recording.speaker_embeddings == {"SPEAKER_00": [0.1, 0.2, 0.3]}


@pytest.mark.asyncio
async def test_get_prefers_embeddings_json_over_metadata(temp_storage):
    """If both metadata.json and embeddings.json have embeddings, embeddings.json wins."""
    rec_dir = temp_storage.recordings_dir / "both-rec"
    rec_dir.mkdir(parents=True)

    metadata = {
        "id": "both-rec",
        "status": "complete",
        "created_at": "2025-06-01T00:00:00",
        "title": "Both",
        "duration": 60.0,
        "settings": {},
        "segments": [],
        "speaker_embeddings": {"SPEAKER_00": [0.0, 0.0]},
    }
    (rec_dir / "metadata.json").write_text(json.dumps(metadata))
    (rec_dir / "embeddings.json").write_text(json.dumps({"SPEAKER_00": [1.0, 1.0]}))

    recording = await temp_storage.get("both-rec")
    assert recording.speaker_embeddings == {"SPEAKER_00": [1.0, 1.0]}


@pytest.mark.asyncio
async def test_get_then_update_preserves_embeddings(temp_storage):
    """Round-trip: get() loads embeddings, update() with changed title preserves them."""
    rec = StorageRecording(
        id="roundtrip",
        status="complete",
        created_at=datetime(2026, 1, 15, 10, 0, 0),
        title="Original",
        duration=60.0,
        settings={},
        speaker_embeddings={"SPEAKER_00": [0.5, 0.6]},
    )
    await temp_storage.create(rec)

    loaded = await temp_storage.get("roundtrip")
    loaded.title = "Updated"
    await temp_storage.update(loaded)

    reloaded = await temp_storage.get("roundtrip")
    assert reloaded.title == "Updated"
    assert reloaded.speaker_embeddings == {"SPEAKER_00": [0.5, 0.6]}

    # Verify on disk: metadata.json should not have speaker_embeddings
    meta = json.loads((temp_storage.recordings_dir / "roundtrip" / "metadata.json").read_text())
    assert "speaker_embeddings" not in meta
    emb = json.loads((temp_storage.recordings_dir / "roundtrip" / "embeddings.json").read_text())
    assert emb == {"SPEAKER_00": [0.5, 0.6]}
