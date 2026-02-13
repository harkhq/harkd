"""Tests for filesystem voice profile storage."""

from datetime import datetime

import pytest

from harkd.exceptions import StorageError, VoiceProfileNotFoundError
from harkd.storage.filesystem.voice_profiles import FilesystemVoiceProfileStorage
from harkd.storage.models import StorageVoiceProfile


@pytest.fixture
def temp_storage(tmp_path):
    """Create a temporary storage instance."""
    return FilesystemVoiceProfileStorage(tmp_path)


@pytest.fixture
def sample_profile():
    """Create a sample voice profile for testing."""
    return StorageVoiceProfile(
        id="prof-123",
        name="Alice",
        created_at=datetime(2026, 1, 10, 14, 20, 0),
        last_used=datetime(2026, 1, 15, 10, 30, 0),
        clips=5,
        total_seconds=120.5,
        confidence=0.85,
        embeddings=[
            {"recording_id": "rec-1", "speaker_id": "SPEAKER_00"},
            {"recording_id": "rec-2", "speaker_id": "SPEAKER_01"},
        ],
    )


@pytest.mark.asyncio
async def test_create_profile(temp_storage, sample_profile):
    """Test creating a new voice profile."""
    created = await temp_storage.create(sample_profile)

    assert created.id == sample_profile.id
    assert created.name == sample_profile.name

    # Verify file was created
    profile_file = temp_storage.profiles_dir / f"{sample_profile.id}.json"
    assert profile_file.exists()


@pytest.mark.asyncio
async def test_create_duplicate_profile(temp_storage, sample_profile):
    """Test creating a profile with duplicate ID fails."""
    await temp_storage.create(sample_profile)

    # Attempting to create again should fail
    with pytest.raises(StorageError) as exc_info:
        await temp_storage.create(sample_profile)

    assert "already exists" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_get_existing_profile(temp_storage, sample_profile):
    """Test getting an existing profile."""
    await temp_storage.create(sample_profile)

    retrieved = await temp_storage.get(sample_profile.id)

    assert retrieved is not None
    assert retrieved.id == sample_profile.id
    assert retrieved.name == sample_profile.name
    assert retrieved.clips == sample_profile.clips
    assert retrieved.total_seconds == sample_profile.total_seconds
    assert retrieved.confidence == sample_profile.confidence
    assert len(retrieved.embeddings) == 2


@pytest.mark.asyncio
async def test_get_nonexistent_profile(temp_storage):
    """Test getting a non-existent profile returns None."""
    result = await temp_storage.get("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_list_profiles_empty(temp_storage):
    """Test listing when no profiles exist."""
    profiles = await temp_storage.list()
    assert profiles == []


@pytest.mark.asyncio
async def test_list_profiles_multiple(temp_storage):
    """Test listing multiple profiles."""
    # Create multiple profiles
    prof1 = StorageVoiceProfile(
        id="prof-1",
        name="Charlie",
        created_at=datetime.now(),
    )
    prof2 = StorageVoiceProfile(
        id="prof-2",
        name="Alice",
        created_at=datetime.now(),
    )
    prof3 = StorageVoiceProfile(
        id="prof-3",
        name="Bob",
        created_at=datetime.now(),
    )

    await temp_storage.create(prof1)
    await temp_storage.create(prof2)
    await temp_storage.create(prof3)

    # List all
    all_profiles = await temp_storage.list()
    assert len(all_profiles) == 3

    # Should be sorted by name (case-insensitive)
    assert all_profiles[0].name == "Alice"
    assert all_profiles[1].name == "Bob"
    assert all_profiles[2].name == "Charlie"


@pytest.mark.asyncio
async def test_update_profile(temp_storage, sample_profile):
    """Test updating an existing profile."""
    await temp_storage.create(sample_profile)

    # Modify the profile
    sample_profile.name = "Alice Updated"
    sample_profile.clips = 10
    sample_profile.total_seconds = 250.0

    updated = await temp_storage.update(sample_profile)
    assert updated.name == "Alice Updated"
    assert updated.clips == 10
    assert updated.total_seconds == 250.0

    # Verify persistence
    retrieved = await temp_storage.get(sample_profile.id)
    assert retrieved.name == "Alice Updated"
    assert retrieved.clips == 10


@pytest.mark.asyncio
async def test_update_nonexistent_profile(temp_storage, sample_profile):
    """Test updating a non-existent profile fails."""
    with pytest.raises(VoiceProfileNotFoundError):
        await temp_storage.update(sample_profile)


@pytest.mark.asyncio
async def test_delete_profile(temp_storage, sample_profile):
    """Test deleting a profile."""
    await temp_storage.create(sample_profile)

    # Verify it exists
    assert await temp_storage.get(sample_profile.id) is not None

    # Delete it
    await temp_storage.delete(sample_profile.id)

    # Verify it's gone
    assert await temp_storage.get(sample_profile.id) is None

    # Verify file is removed
    profile_file = temp_storage.profiles_dir / f"{sample_profile.id}.json"
    assert not profile_file.exists()


@pytest.mark.asyncio
async def test_delete_nonexistent_profile(temp_storage):
    """Test deleting a non-existent profile fails."""
    with pytest.raises(VoiceProfileNotFoundError):
        await temp_storage.delete("nonexistent-id")


@pytest.mark.asyncio
async def test_json_serialization_roundtrip(temp_storage, sample_profile):
    """Test that profiles survive JSON serialization/deserialization."""
    await temp_storage.create(sample_profile)
    retrieved = await temp_storage.get(sample_profile.id)

    # All fields should match
    assert retrieved.id == sample_profile.id
    assert retrieved.name == sample_profile.name
    assert retrieved.created_at == sample_profile.created_at
    assert retrieved.last_used == sample_profile.last_used
    assert retrieved.clips == sample_profile.clips
    assert retrieved.total_seconds == sample_profile.total_seconds
    assert retrieved.confidence == sample_profile.confidence
    assert retrieved.embeddings == sample_profile.embeddings
