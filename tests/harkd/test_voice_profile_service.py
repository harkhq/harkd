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
async def test_remove_embedding(profile_service, temp_storage):
    """Test removing an embedding from a profile."""
    from harkd.storage.models import StorageVoiceProfile

    now = datetime.now()
    storage_profile = StorageVoiceProfile(
        id="prof-rm",
        name="Alice",
        created_at=now,
        clips=2,
        total_seconds=120.0,
        confidence=0.4,
        embeddings=[
            {
                "recording_id": "rec-1",
                "speaker_id": "SPEAKER_00",
                "timestamp": now.isoformat(),
                "vector": [0.1, 0.2],
                "audio_duration": 45.0,
            },
            {
                "recording_id": "rec-2",
                "speaker_id": "SPEAKER_01",
                "timestamp": now.isoformat(),
                "vector": [0.3, 0.4],
                "audio_duration": 75.0,
            },
        ],
    )
    await temp_storage.create(storage_profile)

    await profile_service.remove_embedding("prof-rm", "rec-1", "SPEAKER_00")

    detail = await profile_service.get_profile_detail("prof-rm")
    assert len(detail.embeddings) == 1
    assert detail.embeddings[0].recording_id == "rec-2"
    assert detail.clips == 1
    assert detail.confidence == min(1.0, 1 * 0.2)


@pytest.mark.asyncio
async def test_remove_embedding_decrements_total_seconds(profile_service, temp_storage):
    """Test that remove_embedding correctly decrements total_seconds."""
    from harkd.storage.models import StorageVoiceProfile

    now = datetime.now()
    storage_profile = StorageVoiceProfile(
        id="prof-secs",
        name="Alice",
        created_at=now,
        clips=2,
        total_seconds=100.0,
        confidence=0.4,
        embeddings=[
            {
                "recording_id": "rec-1",
                "speaker_id": "SPEAKER_00",
                "timestamp": now.isoformat(),
                "vector": [0.1],
                "audio_duration": 40.0,
            },
            {
                "recording_id": "rec-2",
                "speaker_id": "SPEAKER_01",
                "timestamp": now.isoformat(),
                "vector": [0.2],
                "audio_duration": 60.0,
            },
        ],
    )
    await temp_storage.create(storage_profile)

    await profile_service.remove_embedding("prof-secs", "rec-1", "SPEAKER_00")

    profile = await profile_service.get_profile("prof-secs")
    assert profile.total_seconds == 60.0
    assert profile.clips == 1


@pytest.mark.asyncio
async def test_remove_embedding_without_audio_duration(profile_service, temp_storage):
    """Test remove_embedding handles legacy embeddings without audio_duration."""
    from harkd.storage.models import StorageVoiceProfile

    now = datetime.now()
    storage_profile = StorageVoiceProfile(
        id="prof-legacy",
        name="Legacy",
        created_at=now,
        clips=1,
        total_seconds=50.0,
        confidence=0.2,
        embeddings=[
            {
                "recording_id": "rec-1",
                "speaker_id": "SPEAKER_00",
                "timestamp": now.isoformat(),
                "vector": [0.1],
                # No audio_duration — legacy embedding
            },
        ],
    )
    await temp_storage.create(storage_profile)

    # Should not crash, total_seconds should not go negative
    await profile_service.remove_embedding("prof-legacy", "rec-1", "SPEAKER_00")

    profile = await profile_service.get_profile("prof-legacy")
    assert profile.clips == 0
    assert profile.total_seconds >= 0  # Doesn't go negative


@pytest.mark.asyncio
async def test_remove_embedding_nonexistent(profile_service, temp_storage):
    """Test removing a non-matching embedding is a no-op."""
    from harkd.storage.models import StorageVoiceProfile

    now = datetime.now()
    storage_profile = StorageVoiceProfile(
        id="prof-noop",
        name="Bob",
        created_at=now,
        clips=1,
        total_seconds=60.0,
        confidence=0.2,
        embeddings=[
            {
                "recording_id": "rec-1",
                "speaker_id": "SPEAKER_00",
                "timestamp": now.isoformat(),
                "vector": [0.1],
            },
        ],
    )
    await temp_storage.create(storage_profile)

    # Remove a non-existent embedding
    await profile_service.remove_embedding("prof-noop", "rec-99", "SPEAKER_99")

    detail = await profile_service.get_profile_detail("prof-noop")
    assert len(detail.embeddings) == 1
    assert detail.clips == 1
    # total_seconds should be unchanged
    assert detail.total_seconds == 60.0


@pytest.mark.asyncio
async def test_remove_embedding_profile_not_found(profile_service):
    """Test removing embedding from nonexistent profile raises error."""
    with pytest.raises(VoiceProfileNotFoundError):
        await profile_service.remove_embedding("nonexistent", "rec-1", "SPEAKER_00")


@pytest.mark.asyncio
async def test_add_embedding_stores_audio_duration(profile_service):
    """Test that add_embedding stores audio_duration in the embedding dict."""
    created = await profile_service.create_profile(VoiceProfileCreate(name="Test"))

    await profile_service.add_embedding(
        profile_id=created.id,
        recording_id="rec-1",
        speaker_label="SPEAKER_00",
        vector=[0.1, 0.2],
        audio_duration=42.5,
    )

    detail = await profile_service.get_profile_detail(created.id)
    assert len(detail.embeddings) == 1
    assert detail.total_seconds == 42.5
    assert detail.clips == 1

    # Verify the audio_duration is stored (check storage directly)
    storage_profile = await profile_service.storage.get(created.id)
    assert storage_profile.embeddings[0]["audio_duration"] == 42.5


@pytest.mark.asyncio
async def test_add_then_remove_embedding_roundtrip(profile_service):
    """Test that add + remove returns profile to correct state."""
    created = await profile_service.create_profile(VoiceProfileCreate(name="Roundtrip"))

    await profile_service.add_embedding(
        profile_id=created.id,
        recording_id="rec-1",
        speaker_label="SPEAKER_00",
        vector=[0.1],
        audio_duration=30.0,
    )
    await profile_service.add_embedding(
        profile_id=created.id,
        recording_id="rec-2",
        speaker_label="SPEAKER_01",
        vector=[0.2],
        audio_duration=20.0,
    )

    profile = await profile_service.get_profile(created.id)
    assert profile.clips == 2
    assert profile.total_seconds == 50.0

    await profile_service.remove_embedding(created.id, "rec-1", "SPEAKER_00")

    profile = await profile_service.get_profile(created.id)
    assert profile.clips == 1
    assert profile.total_seconds == 20.0
    assert profile.confidence == 0.2

    await profile_service.remove_embedding(created.id, "rec-2", "SPEAKER_01")

    profile = await profile_service.get_profile(created.id)
    assert profile.clips == 0
    assert profile.total_seconds == 0.0
    assert profile.confidence == 0.0


@pytest.mark.asyncio
async def test_get_known_embeddings(profile_service, temp_storage):
    """Test get_known_embeddings returns averaged embeddings for profiles with vectors."""
    from harkd.storage.models import StorageVoiceProfile

    now = datetime.now()

    # Profile with 2 embeddings
    prof_alice = StorageVoiceProfile(
        id="prof-alice",
        name="Alice",
        created_at=now,
        clips=2,
        total_seconds=100.0,
        confidence=0.4,
        embeddings=[
            {
                "recording_id": "rec-1",
                "speaker_id": "SPEAKER_00",
                "timestamp": now.isoformat(),
                "vector": [1.0, 0.0, 0.0],
                "audio_duration": 50.0,
            },
            {
                "recording_id": "rec-2",
                "speaker_id": "SPEAKER_01",
                "timestamp": now.isoformat(),
                "vector": [0.0, 1.0, 0.0],
                "audio_duration": 50.0,
            },
        ],
    )
    await temp_storage.create(prof_alice)

    # Profile with no embeddings (should be excluded)
    prof_empty = StorageVoiceProfile(
        id="prof-empty",
        name="Empty",
        created_at=now,
        clips=0,
        total_seconds=0,
        confidence=0,
        embeddings=[],
    )
    await temp_storage.create(prof_empty)

    result = await profile_service.get_known_embeddings()

    assert "Alice" in result
    assert "Empty" not in result

    avg_vec, profile_id = result["Alice"]
    assert profile_id == "prof-alice"
    # Average of [1,0,0] and [0,1,0] is [0.5, 0.5, 0.0]
    assert len(avg_vec) == 3
    assert abs(avg_vec[0] - 0.5) < 1e-6
    assert abs(avg_vec[1] - 0.5) < 1e-6
    assert abs(avg_vec[2] - 0.0) < 1e-6


@pytest.mark.asyncio
async def test_get_known_embeddings_empty(profile_service):
    """Test get_known_embeddings with no profiles returns empty dict."""
    result = await profile_service.get_known_embeddings()
    assert result == {}


@pytest.mark.asyncio
async def test_get_known_embeddings_skips_embeddings_without_vector(profile_service, temp_storage):
    """Test that embeddings without a vector field are skipped in averaging."""
    from harkd.storage.models import StorageVoiceProfile

    now = datetime.now()
    profile = StorageVoiceProfile(
        id="prof-legacy",
        name="Legacy",
        created_at=now,
        clips=2,
        total_seconds=100.0,
        confidence=0.4,
        embeddings=[
            {
                "recording_id": "rec-1",
                "speaker_id": "SPEAKER_00",
                "timestamp": now.isoformat(),
                # No vector field — legacy embedding
            },
            {
                "recording_id": "rec-2",
                "speaker_id": "SPEAKER_01",
                "timestamp": now.isoformat(),
                "vector": [0.5, 0.5, 0.5],
                "audio_duration": 50.0,
            },
        ],
    )
    await temp_storage.create(profile)

    result = await profile_service.get_known_embeddings()

    assert "Legacy" in result
    avg_vec, _ = result["Legacy"]
    # Only one valid vector, so average is that vector itself
    assert abs(avg_vec[0] - 0.5) < 1e-6


@pytest.mark.asyncio
async def test_get_known_embeddings_single_embedding(profile_service, temp_storage):
    """Test get_known_embeddings with a profile that has exactly one embedding."""
    from harkd.storage.models import StorageVoiceProfile

    now = datetime.now()
    profile = StorageVoiceProfile(
        id="prof-single",
        name="Single",
        created_at=now,
        clips=1,
        total_seconds=30.0,
        confidence=0.2,
        embeddings=[
            {
                "recording_id": "rec-1",
                "speaker_id": "SPEAKER_00",
                "timestamp": now.isoformat(),
                "vector": [0.3, 0.4, 0.5],
                "audio_duration": 30.0,
            },
        ],
    )
    await temp_storage.create(profile)

    result = await profile_service.get_known_embeddings()

    assert "Single" in result
    avg_vec, pid = result["Single"]
    assert pid == "prof-single"
    # Average of one vector is itself
    assert abs(avg_vec[0] - 0.3) < 1e-6
    assert abs(avg_vec[1] - 0.4) < 1e-6
    assert abs(avg_vec[2] - 0.5) < 1e-6


@pytest.mark.asyncio
async def test_get_known_embeddings_duplicate_profile_names(profile_service, temp_storage):
    """Test that duplicate profile names cause last-writer-wins in the dict.

    This is a known limitation — if two profiles share a name,
    only one appears in the result (dict key collision).
    """
    from harkd.storage.models import StorageVoiceProfile

    now = datetime.now()
    for i, pid in enumerate(["prof-dup-1", "prof-dup-2"]):
        profile = StorageVoiceProfile(
            id=pid,
            name="Alice",  # Same name!
            created_at=now,
            clips=1,
            total_seconds=30.0,
            confidence=0.2,
            embeddings=[
                {
                    "recording_id": f"rec-{i}",
                    "speaker_id": "SPEAKER_00",
                    "timestamp": now.isoformat(),
                    "vector": [float(i), 0.0, 0.0],
                    "audio_duration": 30.0,
                },
            ],
        )
        await temp_storage.create(profile)

    result = await profile_service.get_known_embeddings()

    # Only one "Alice" key — the other is silently lost
    assert "Alice" in result
    assert len(result) == 1


@pytest.mark.asyncio
async def test_get_known_embeddings_empty_vector_list_skipped(profile_service, temp_storage):
    """Test that an embedding with vector=[] is treated as missing."""
    from harkd.storage.models import StorageVoiceProfile

    now = datetime.now()
    profile = StorageVoiceProfile(
        id="prof-emptyvec",
        name="EmptyVec",
        created_at=now,
        clips=1,
        total_seconds=30.0,
        confidence=0.2,
        embeddings=[
            {
                "recording_id": "rec-1",
                "speaker_id": "SPEAKER_00",
                "timestamp": now.isoformat(),
                "vector": [],  # Empty list
                "audio_duration": 30.0,
            },
        ],
    )
    await temp_storage.create(profile)

    result = await profile_service.get_known_embeddings()

    # vector=[] is falsy — profile should be excluded
    assert "EmptyVec" not in result


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
