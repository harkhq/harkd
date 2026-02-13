"""Tests for storage models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from harkd.storage.models import StorageRecording, StorageVoiceProfile


def test_storage_recording_minimal():
    """Test StorageRecording with minimal required fields."""
    recording = StorageRecording(
        id="test-123",
        status="recording",
        created_at=datetime.now(),
        title="Test Recording",
        duration=0.0,
        settings={},
    )
    assert recording.id == "test-123"
    assert recording.status == "recording"
    assert recording.duration == 0.0
    assert recording.speakers == []
    assert recording.segments == []
    assert recording.tags == []


def test_storage_recording_complete():
    """Test StorageRecording with all fields."""
    now = datetime.now()
    recording = StorageRecording(
        id="test-456",
        status="complete",
        created_at=now,
        title="Complete Recording",
        duration=120.5,
        input_source="mic",
        model="base",
        language="en",
        language_confidence=0.98,
        diarized=True,
        speakers=["Alice", "Bob"],
        segments=[{"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "Alice"}],
        transcript="Hello world",
        settings={"model": "base"},
    )
    assert recording.status == "complete"
    assert recording.duration == 120.5
    assert len(recording.speakers) == 2
    assert len(recording.segments) == 1
    assert recording.transcript == "Hello world"


def test_storage_recording_status_validation():
    """Test StorageRecording status must be valid literal."""
    # Valid statuses
    for status in ["recording", "processing", "complete", "error"]:
        recording = StorageRecording(
            id="test",
            status=status,
            created_at=datetime.now(),
            title="Test",
            duration=0.0,
            settings={},
        )
        assert recording.status == status

    # Invalid status
    with pytest.raises(ValidationError):
        StorageRecording(
            id="test",
            status="invalid",
            created_at=datetime.now(),
            title="Test",
            duration=0.0,
            settings={},
        )


def test_storage_recording_duration_validation():
    """Test StorageRecording duration must be non-negative."""
    # Valid duration
    StorageRecording(
        id="test",
        status="complete",
        created_at=datetime.now(),
        title="Test",
        duration=0.0,
        settings={},
    )

    # Negative duration should fail
    with pytest.raises(ValidationError):
        StorageRecording(
            id="test",
            status="complete",
            created_at=datetime.now(),
            title="Test",
            duration=-1.0,
            settings={},
        )


def test_storage_recording_audio_level_validation():
    """Test StorageRecording audio_level must be 0-1."""
    # Valid levels
    for level in [0.0, 0.5, 1.0]:
        recording = StorageRecording(
            id="test",
            status="recording",
            created_at=datetime.now(),
            title="Test",
            duration=0.0,
            audio_level=level,
            settings={},
        )
        assert recording.audio_level == level

    # Out of range
    with pytest.raises(ValidationError):
        StorageRecording(
            id="test",
            status="recording",
            created_at=datetime.now(),
            title="Test",
            duration=0.0,
            audio_level=1.5,
            settings={},
        )


def test_storage_recording_json_serialization():
    """Test StorageRecording can be serialized to/from JSON."""
    original = StorageRecording(
        id="test-789",
        status="complete",
        created_at=datetime(2026, 1, 15, 10, 30, 0),
        title="JSON Test",
        duration=60.0,
        speakers=["Alice"],
        settings={"model": "base"},
    )

    # Serialize to dict (JSON-compatible)
    data = original.model_dump(mode="json")
    assert isinstance(data, dict)
    assert data["id"] == "test-789"
    assert isinstance(data["created_at"], str)  # Datetime serialized to string

    # Deserialize back
    restored = StorageRecording.model_validate(data)
    assert restored.id == original.id
    assert restored.title == original.title
    assert restored.duration == original.duration


def test_storage_voice_profile_minimal():
    """Test StorageVoiceProfile with minimal fields."""
    profile = StorageVoiceProfile(
        id="prof-123",
        name="Alice",
        created_at=datetime.now(),
    )
    assert profile.id == "prof-123"
    assert profile.name == "Alice"
    assert profile.last_used is None
    assert profile.clips == 0
    assert profile.total_seconds == 0.0
    assert profile.confidence == 0.0
    assert profile.embeddings == []


def test_storage_voice_profile_complete():
    """Test StorageVoiceProfile with all fields."""
    now = datetime.now()
    profile = StorageVoiceProfile(
        id="prof-456",
        name="Bob",
        created_at=now,
        last_used=now,
        clips=5,
        total_seconds=120.5,
        confidence=0.85,
        embeddings=[{"recording_id": "rec-1", "speaker_id": "SPEAKER_00"}],
    )
    assert profile.clips == 5
    assert profile.total_seconds == 120.5
    assert profile.confidence == 0.85
    assert len(profile.embeddings) == 1


def test_storage_voice_profile_validation():
    """Test StorageVoiceProfile field validation."""
    # clips must be non-negative
    with pytest.raises(ValidationError):
        StorageVoiceProfile(
            id="prof",
            name="Test",
            created_at=datetime.now(),
            clips=-1,
        )

    # total_seconds must be non-negative
    with pytest.raises(ValidationError):
        StorageVoiceProfile(
            id="prof",
            name="Test",
            created_at=datetime.now(),
            total_seconds=-1.0,
        )

    # confidence must be 0-1
    with pytest.raises(ValidationError):
        StorageVoiceProfile(
            id="prof",
            name="Test",
            created_at=datetime.now(),
            confidence=1.5,
        )


def test_storage_models_ignore_extra_fields():
    """Test that storage models ignore extra fields for forward compatibility."""
    # StorageRecording with extra field - should not raise, but field is dropped
    recording = StorageRecording(
        id="test",
        status="complete",
        created_at=datetime.now(),
        title="Test",
        duration=0.0,
        settings={},
        future_field="future_value",  # Extra field - silently ignored
    )
    assert not hasattr(recording, "future_field")

    # StorageVoiceProfile with extra field - should not raise, but field is dropped
    profile = StorageVoiceProfile(
        id="prof",
        name="Test",
        created_at=datetime.now(),
        new_metric=42,  # Extra field - silently ignored
    )
    assert not hasattr(profile, "new_metric")
