"""Tests for voice profile service."""

from datetime import datetime

import pytest

from harkd.api.models.voice_profile import VoiceProfileCreate
from harkd.exceptions import VoiceProfileNotFoundError
from harkd.services.voice_profile_service import VoiceProfileService
from harkd.storage.filesystem.voice_profiles import FilesystemVoiceProfileStorage


@pytest.fixture
def temp_storage(tmp_path):
    """Create temporary voice profile storage."""
    return FilesystemVoiceProfileStorage(tmp_path)


@pytest.fixture
def profile_service(temp_storage):
    """Create voice profile service with temporary storage."""
    return VoiceProfileService(temp_storage)


@pytest.mark.asyncio
async def test_create_profile(profile_service):
    """Test creating a new profile."""
    create = VoiceProfileCreate(name="Alice")
    profile = await profile_service.create_profile(create)

    assert profile.name == "Alice"
    assert profile.id.startswith("profile-")
    assert profile.clips == 0
    assert profile.total_seconds == 0
    assert profile.confidence == 0
    assert profile.last_used is None
    assert isinstance(profile.created_at, datetime)


@pytest.mark.asyncio
async def test_create_profile_generates_unique_id(profile_service):
    """Test that each created profile gets a unique ID."""
    profile1 = await profile_service.create_profile(VoiceProfileCreate(name="Alice"))
    profile2 = await profile_service.create_profile(VoiceProfileCreate(name="Bob"))

    assert profile1.id != profile2.id


@pytest.mark.asyncio
async def test_get_profile(profile_service):
    """Test getting an existing profile."""
    # Create profile
    created = await profile_service.create_profile(VoiceProfileCreate(name="Alice"))

    # Get profile
    retrieved = await profile_service.get_profile(created.id)

    assert retrieved.id == created.id
    assert retrieved.name == "Alice"
    assert retrieved.clips == 0


@pytest.mark.asyncio
async def test_get_profile_not_found(profile_service):
    """Test getting non-existent profile raises error."""
    with pytest.raises(VoiceProfileNotFoundError) as exc_info:
        await profile_service.get_profile("nonexistent-id")

    assert "nonexistent-id" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_get_profile_detail(profile_service):
    """Test getting detailed profile with embeddings."""
    # Create profile
    created = await profile_service.create_profile(VoiceProfileCreate(name="Alice"))

    # Get detail
    detail = await profile_service.get_profile_detail(created.id)

    assert detail.id == created.id
    assert detail.name == "Alice"
    assert detail.embeddings == []  # No embeddings yet


@pytest.mark.asyncio
async def test_get_profile_detail_not_found(profile_service):
    """Test getting detail for non-existent profile raises error."""
    with pytest.raises(VoiceProfileNotFoundError):
        await profile_service.get_profile_detail("nonexistent-id")


@pytest.mark.asyncio
async def test_list_profiles_empty(profile_service):
    """Test listing when no profiles exist."""
    response = await profile_service.list_profiles()
    assert response.profiles == []


@pytest.mark.asyncio
async def test_list_profiles_multiple(profile_service):
    """Test listing multiple profiles."""
    # Create profiles
    await profile_service.create_profile(VoiceProfileCreate(name="Charlie"))
    await profile_service.create_profile(VoiceProfileCreate(name="Alice"))
    await profile_service.create_profile(VoiceProfileCreate(name="Bob"))

    # List profiles
    response = await profile_service.list_profiles()

    assert len(response.profiles) == 3
    # Should be sorted by name (storage layer sorts)
    names = [p.name for p in response.profiles]
    assert names == ["Alice", "Bob", "Charlie"]


@pytest.mark.asyncio
async def test_delete_profile(profile_service):
    """Test deleting a profile."""
    # Create profile
    created = await profile_service.create_profile(VoiceProfileCreate(name="Alice"))

    # Verify it exists
    profile = await profile_service.get_profile(created.id)
    assert profile.name == "Alice"

    # Delete it
    await profile_service.delete_profile(created.id)

    # Verify it's gone
    with pytest.raises(VoiceProfileNotFoundError):
        await profile_service.get_profile(created.id)


@pytest.mark.asyncio
async def test_delete_profile_not_found(profile_service):
    """Test deleting non-existent profile raises error."""
    with pytest.raises(VoiceProfileNotFoundError):
        await profile_service.delete_profile("nonexistent-id")


@pytest.mark.asyncio
async def test_profile_detail_with_embeddings(profile_service, temp_storage):
    """Test profile detail includes embeddings."""
    from harkd.storage.models import StorageVoiceProfile

    # Create profile with embeddings directly in storage
    now = datetime.now()
    storage_profile = StorageVoiceProfile(
        id="prof-001",
        name="Alice",
        created_at=now,
        clips=2,
        total_seconds=120.5,
        confidence=0.85,
        embeddings=[
            {
                "recording_id": "rec-1",
                "speaker_id": "SPEAKER_00",
                "timestamp": now.isoformat(),
            },
            {
                "recording_id": "rec-2",
                "speaker_id": "SPEAKER_01",
                "timestamp": now.isoformat(),
            },
        ],
    )
    await temp_storage.create(storage_profile)

    # Get detail through service
    detail = await profile_service.get_profile_detail("prof-001")

    assert len(detail.embeddings) == 2
    assert detail.embeddings[0].recording_id == "rec-1"
    assert detail.embeddings[0].speaker_id == "SPEAKER_00"
    assert detail.embeddings[1].recording_id == "rec-2"
    assert detail.clips == 2
    assert detail.total_seconds == 120.5
    assert detail.confidence == 0.85


@pytest.mark.asyncio
async def test_list_reflects_deletions(profile_service):
    """Test that list reflects profile deletions."""
    # Create profiles
    prof1 = await profile_service.create_profile(VoiceProfileCreate(name="Alice"))
    prof2 = await profile_service.create_profile(VoiceProfileCreate(name="Bob"))

    # Should have 2
    response = await profile_service.list_profiles()
    assert len(response.profiles) == 2

    # Delete one
    await profile_service.delete_profile(prof1.id)

    # Should have 1
    response = await profile_service.list_profiles()
    assert len(response.profiles) == 1
    assert response.profiles[0].id == prof2.id
