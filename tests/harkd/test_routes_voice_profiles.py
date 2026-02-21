"""Tests for voice profile API routes."""

import json

import pytest
from fastapi.testclient import TestClient

from harkd.api.app import create_app
from harkd.api.routes.voice_profiles import router
from harkd.config import HarkdSettings, StorageSettings


@pytest.fixture
def test_settings(tmp_path):
    """Create test settings."""
    return HarkdSettings(storage=StorageSettings(base_path=tmp_path))


@pytest.fixture
def app(test_settings):
    """Create test FastAPI app with voice profile routes."""
    from harkd.config import get_settings

    app = create_app(test_settings)
    app.include_router(router)

    # Override the settings dependency to use test settings
    app.dependency_overrides[get_settings] = lambda: test_settings
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


def _create_recording_in_storage(settings, recording_id, **kwargs):
    """Helper to create a recording directly in storage.

    Writes speaker_embeddings to a separate embeddings.json file,
    matching the on-disk format used by FilesystemRecordingStorage.
    """
    rec_dir = settings.storage.base_path / "recordings" / recording_id
    rec_dir.mkdir(parents=True, exist_ok=True)

    speaker_embeddings = kwargs.get("speaker_embeddings")

    metadata = {
        "id": recording_id,
        "status": kwargs.get("status", "complete"),
        "created_at": kwargs.get("created_at", "2025-01-01T00:00:00"),
        "title": kwargs.get("title", "Test Recording"),
        "duration": kwargs.get("duration", 60.0),
        "settings": {},
        "segments": kwargs.get("segments", []),
        "speakers": kwargs.get("speakers", []),
        "speaker_profiles": kwargs.get("speaker_profiles"),
        "diarized": kwargs.get("diarized", True),
    }
    with open(rec_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    if speaker_embeddings is not None:
        with open(rec_dir / "embeddings.json", "w") as f:
            json.dump(speaker_embeddings, f)


def test_list_profiles_empty(client):
    """Test listing when no profiles exist."""
    response = client.get("/api/v1/voice-profiles")
    assert response.status_code == 200

    data = response.json()
    assert data["profiles"] == []


def test_create_profile(client):
    """Test creating a new profile."""
    create_data = {"name": "Alice"}
    response = client.post("/api/v1/voice-profiles", json=create_data)
    assert response.status_code == 201

    data = response.json()
    assert data["name"] == "Alice"
    assert "id" in data
    assert data["id"].startswith("profile-")
    assert data["clips"] == 0
    assert data["total_seconds"] == 0
    assert data["confidence"] == 0
    assert data["last_used"] is None


def test_create_profile_name_validation(client):
    """Test profile name validation."""
    # Empty name should fail
    response = client.post("/api/v1/voice-profiles", json={"name": ""})
    assert response.status_code == 422

    # Name too long should fail
    long_name = "x" * 101
    response = client.post("/api/v1/voice-profiles", json={"name": long_name})
    assert response.status_code == 422


def test_get_profile(client):
    """Test getting a profile by ID."""
    # Create profile
    create_response = client.post("/api/v1/voice-profiles", json={"name": "Alice"})
    profile_id = create_response.json()["id"]

    # Get profile
    response = client.get(f"/api/v1/voice-profiles/{profile_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == profile_id
    assert data["name"] == "Alice"
    assert "embeddings" in data
    assert data["embeddings"] == []


def test_get_profile_not_found(client):
    """Test getting non-existent profile returns 404."""
    response = client.get("/api/v1/voice-profiles/nonexistent-id")
    assert response.status_code == 404

    data = response.json()
    assert data["error"]["code"] == "VOICE_PROFILE_NOT_FOUND"
    assert "nonexistent-id" in data["error"]["message"]


def test_list_profiles_multiple(client):
    """Test listing multiple profiles."""
    # Create profiles
    client.post("/api/v1/voice-profiles", json={"name": "Charlie"})
    client.post("/api/v1/voice-profiles", json={"name": "Alice"})
    client.post("/api/v1/voice-profiles", json={"name": "Bob"})

    # List profiles
    response = client.get("/api/v1/voice-profiles")
    assert response.status_code == 200

    data = response.json()
    assert len(data["profiles"]) == 3

    # Should be sorted by name
    names = [p["name"] for p in data["profiles"]]
    assert names == ["Alice", "Bob", "Charlie"]


def test_delete_profile(client):
    """Test deleting a profile."""
    # Create profile
    create_response = client.post("/api/v1/voice-profiles", json={"name": "Alice"})
    profile_id = create_response.json()["id"]

    # Verify it exists
    response = client.get(f"/api/v1/voice-profiles/{profile_id}")
    assert response.status_code == 200

    # Delete it
    response = client.delete(f"/api/v1/voice-profiles/{profile_id}")
    assert response.status_code == 204

    # Verify it's gone
    response = client.get(f"/api/v1/voice-profiles/{profile_id}")
    assert response.status_code == 404


def test_delete_profile_not_found(client):
    """Test deleting non-existent profile returns 404."""
    response = client.delete("/api/v1/voice-profiles/nonexistent-id")
    assert response.status_code == 404

    data = response.json()
    assert data["error"]["code"] == "VOICE_PROFILE_NOT_FOUND"


def test_list_reflects_creation_and_deletion(client):
    """Test that list reflects profile creation and deletion."""
    # Create profiles
    prof1_response = client.post("/api/v1/voice-profiles", json={"name": "Alice"})
    client.post("/api/v1/voice-profiles", json={"name": "Bob"})
    prof1_id = prof1_response.json()["id"]

    # Should have 2
    response = client.get("/api/v1/voice-profiles")
    assert len(response.json()["profiles"]) == 2

    # Delete one
    client.delete(f"/api/v1/voice-profiles/{prof1_id}")

    # Should have 1
    response = client.get("/api/v1/voice-profiles")
    data = response.json()
    assert len(data["profiles"]) == 1
    assert data["profiles"][0]["name"] == "Bob"


def test_profile_detail_response_includes_embeddings(client):
    """Test that profile detail response includes embeddings field."""
    # Create profile
    create_response = client.post("/api/v1/voice-profiles", json={"name": "Alice"})
    profile_id = create_response.json()["id"]

    # Get profile detail
    response = client.get(f"/api/v1/voice-profiles/{profile_id}")
    assert response.status_code == 200

    data = response.json()
    # Should have all VoiceProfile fields plus embeddings
    assert "id" in data
    assert "name" in data
    assert "created_at" in data
    assert "last_used" in data
    assert "clips" in data
    assert "total_seconds" in data
    assert "confidence" in data
    assert "embeddings" in data


def test_create_multiple_profiles_unique_ids(client):
    """Test that each created profile gets a unique ID."""
    response1 = client.post("/api/v1/voice-profiles", json={"name": "Alice"})
    response2 = client.post("/api/v1/voice-profiles", json={"name": "Bob"})

    id1 = response1.json()["id"]
    id2 = response2.json()["id"]

    assert id1 != id2
    assert id1.startswith("profile-")
    assert id2.startswith("profile-")


def test_profile_routes_response_models(client):
    """Test that all routes return correct response models."""
    # Create profile - should return VoiceProfile (no embeddings in list)
    create_response = client.post("/api/v1/voice-profiles", json={"name": "Alice"})
    assert create_response.status_code == 201
    create_data = create_response.json()
    assert "embeddings" not in create_data  # VoiceProfile, not VoiceProfileDetail

    # List - should return VoiceProfileListResponse
    list_response = client.get("/api/v1/voice-profiles")
    list_data = list_response.json()
    assert "profiles" in list_data
    assert isinstance(list_data["profiles"], list)

    # Get by ID - should return VoiceProfileDetail (with embeddings)
    profile_id = create_data["id"]
    get_response = client.get(f"/api/v1/voice-profiles/{profile_id}")
    get_data = get_response.json()
    assert "embeddings" in get_data  # VoiceProfileDetail


# --- Unassigned speakers tests ---


def test_list_unassigned_speakers_empty(client):
    """Test listing unassigned speakers with no recordings."""
    response = client.get("/api/v1/voice-profiles/unassigned-speakers")
    assert response.status_code == 200

    data = response.json()
    assert data["speakers"] == []


def test_list_unassigned_speakers_with_data(client, test_settings):
    """Test listing unassigned speakers from diarized recordings."""
    _create_recording_in_storage(
        test_settings,
        "rec-1",
        speaker_embeddings={"SPEAKER_00": [0.1, 0.2], "SPEAKER_01": [0.3, 0.4]},
        segments=[
            {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "text": "World", "speaker": "SPEAKER_01"},
        ],
        speakers=["SPEAKER_00", "SPEAKER_01"],
    )

    response = client.get("/api/v1/voice-profiles/unassigned-speakers")
    assert response.status_code == 200

    data = response.json()
    assert len(data["speakers"]) == 2

    labels = {s["speaker_label"] for s in data["speakers"]}
    assert labels == {"SPEAKER_00", "SPEAKER_01"}

    # Each should have segments
    for speaker in data["speakers"]:
        assert len(speaker["segments"]) > 0


def test_list_unassigned_speakers_excludes_assigned(client, test_settings):
    """Test that speakers linked to valid profiles are excluded."""
    # Create a profile first
    create_resp = client.post("/api/v1/voice-profiles", json={"name": "Alice"})
    profile_id = create_resp.json()["id"]

    _create_recording_in_storage(
        test_settings,
        "rec-2",
        speaker_embeddings={"SPEAKER_00": [0.1], "SPEAKER_01": [0.2]},
        speaker_profiles={"SPEAKER_00": profile_id},
        segments=[
            {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "text": "World", "speaker": "SPEAKER_01"},
        ],
        speakers=["SPEAKER_00", "SPEAKER_01"],
    )

    response = client.get("/api/v1/voice-profiles/unassigned-speakers")
    assert response.status_code == 200

    data = response.json()
    assert len(data["speakers"]) == 1
    assert data["speakers"][0]["speaker_label"] == "SPEAKER_01"


# --- Profile clips tests ---


def test_list_profile_clips_with_data(client, test_settings):
    """Test getting clips for a profile with embeddings."""
    # Create profile
    create_resp = client.post("/api/v1/voice-profiles", json={"name": "Alice"})
    profile_id = create_resp.json()["id"]

    # Create recording with speaker data
    _create_recording_in_storage(
        test_settings,
        "rec-clip",
        speaker_embeddings={"SPEAKER_00": [0.1, 0.2]},
        segments=[
            {"start": 0.0, "end": 5.0, "text": "Hello there", "speaker": "SPEAKER_00"},
            {"start": 10.0, "end": 15.0, "text": "More speech", "speaker": "SPEAKER_00"},
        ],
        speakers=["SPEAKER_00"],
    )

    # Assign the clip
    assign_resp = client.post(
        f"/api/v1/voice-profiles/{profile_id}/clips",
        json={"recording_id": "rec-clip", "speaker_label": "SPEAKER_00"},
    )
    assert assign_resp.status_code == 201

    # Get clips
    response = client.get(f"/api/v1/voice-profiles/{profile_id}/clips")
    assert response.status_code == 200

    data = response.json()
    assert data["profile_id"] == profile_id
    assert data["profile_name"] == "Alice"
    assert len(data["clips"]) == 1
    assert data["clips"][0]["recording_id"] == "rec-clip"
    assert len(data["clips"][0]["segments"]) == 2


def test_assign_clip_success(client, test_settings):
    """Test assigning a speaker clip to a profile."""
    # Create profile
    create_resp = client.post("/api/v1/voice-profiles", json={"name": "Bob"})
    profile_id = create_resp.json()["id"]

    # Create recording
    _create_recording_in_storage(
        test_settings,
        "rec-assign",
        speaker_embeddings={"SPEAKER_00": [0.1, 0.2]},
        segments=[
            {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"},
        ],
        speakers=["SPEAKER_00"],
    )

    # Assign
    response = client.post(
        f"/api/v1/voice-profiles/{profile_id}/clips",
        json={"recording_id": "rec-assign", "speaker_label": "SPEAKER_00"},
    )
    assert response.status_code == 201

    data = response.json()
    assert data["clips"] == 1

    # Verify recording speaker_profiles was updated
    import json as json_mod

    rec_dir = test_settings.storage.base_path / "recordings" / "rec-assign"
    with open(rec_dir / "metadata.json") as f:
        rec_data = json_mod.load(f)
    assert rec_data["speaker_profiles"]["SPEAKER_00"] == profile_id


def test_assign_clip_no_embedding(client, test_settings):
    """Test 422 when speaker embedding doesn't exist."""
    # Create profile
    create_resp = client.post("/api/v1/voice-profiles", json={"name": "Charlie"})
    profile_id = create_resp.json()["id"]

    # Create recording without embeddings
    _create_recording_in_storage(
        test_settings,
        "rec-no-emb",
        speaker_embeddings=None,
        segments=[],
        speakers=[],
    )

    response = client.post(
        f"/api/v1/voice-profiles/{profile_id}/clips",
        json={"recording_id": "rec-no-emb", "speaker_label": "SPEAKER_00"},
    )
    assert response.status_code == 422


def test_remove_clip_success(client, test_settings):
    """Test removing a clip from a profile."""
    # Create profile
    create_resp = client.post("/api/v1/voice-profiles", json={"name": "Diana"})
    profile_id = create_resp.json()["id"]

    # Create recording
    _create_recording_in_storage(
        test_settings,
        "rec-remove",
        speaker_embeddings={"SPEAKER_00": [0.1]},
        segments=[
            {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"},
        ],
        speakers=["SPEAKER_00"],
    )

    # Assign
    client.post(
        f"/api/v1/voice-profiles/{profile_id}/clips",
        json={"recording_id": "rec-remove", "speaker_label": "SPEAKER_00"},
    )

    # Verify clips count is 1
    profile_resp = client.get(f"/api/v1/voice-profiles/{profile_id}")
    assert profile_resp.json()["clips"] == 1

    # Remove
    response = client.delete(f"/api/v1/voice-profiles/{profile_id}/clips/rec-remove/SPEAKER_00")
    assert response.status_code == 204

    # Verify clips count is 0
    profile_resp = client.get(f"/api/v1/voice-profiles/{profile_id}")
    assert profile_resp.json()["clips"] == 0

    # Verify recording speaker_profiles was cleaned
    import json as json_mod

    rec_dir = test_settings.storage.base_path / "recordings" / "rec-remove"
    with open(rec_dir / "metadata.json") as f:
        rec_data = json_mod.load(f)
    assert "SPEAKER_00" not in (rec_data.get("speaker_profiles") or {})


# --- Edge case: duplicate and reassignment tests ---


def test_assign_clip_duplicate_returns_409(client, test_settings):
    """Test that assigning the same speaker to the same profile twice returns 409."""
    # Create profile
    create_resp = client.post("/api/v1/voice-profiles", json={"name": "Eve"})
    profile_id = create_resp.json()["id"]

    # Create recording
    _create_recording_in_storage(
        test_settings,
        "rec-dup",
        speaker_embeddings={"SPEAKER_00": [0.1]},
        segments=[
            {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"},
        ],
        speakers=["SPEAKER_00"],
    )

    # First assignment — success
    resp1 = client.post(
        f"/api/v1/voice-profiles/{profile_id}/clips",
        json={"recording_id": "rec-dup", "speaker_label": "SPEAKER_00"},
    )
    assert resp1.status_code == 201

    # Second assignment — duplicate, should get 409
    resp2 = client.post(
        f"/api/v1/voice-profiles/{profile_id}/clips",
        json={"recording_id": "rec-dup", "speaker_label": "SPEAKER_00"},
    )
    assert resp2.status_code == 409
    assert resp2.json()["detail"]["error"]["code"] == "ALREADY_ASSIGNED"

    # Verify profile still has clips=1 (not 2)
    profile_resp = client.get(f"/api/v1/voice-profiles/{profile_id}")
    assert profile_resp.json()["clips"] == 1


def test_reassign_speaker_to_different_profile(client, test_settings):
    """Test reassigning a speaker from profile-A to profile-B."""
    # Create two profiles
    resp_a = client.post("/api/v1/voice-profiles", json={"name": "Profile A"})
    profile_a_id = resp_a.json()["id"]
    resp_b = client.post("/api/v1/voice-profiles", json={"name": "Profile B"})
    profile_b_id = resp_b.json()["id"]

    # Create recording
    _create_recording_in_storage(
        test_settings,
        "rec-reassign",
        speaker_embeddings={"SPEAKER_00": [0.1, 0.2]},
        segments=[
            {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"},
        ],
        speakers=["SPEAKER_00"],
    )

    # Assign to profile A
    resp1 = client.post(
        f"/api/v1/voice-profiles/{profile_a_id}/clips",
        json={"recording_id": "rec-reassign", "speaker_label": "SPEAKER_00"},
    )
    assert resp1.status_code == 201

    profile_a = client.get(f"/api/v1/voice-profiles/{profile_a_id}").json()
    assert profile_a["clips"] == 1

    # Reassign to profile B — should succeed and remove from A
    resp2 = client.post(
        f"/api/v1/voice-profiles/{profile_b_id}/clips",
        json={"recording_id": "rec-reassign", "speaker_label": "SPEAKER_00"},
    )
    assert resp2.status_code == 201

    # Profile A should now have 0 clips
    profile_a = client.get(f"/api/v1/voice-profiles/{profile_a_id}").json()
    assert profile_a["clips"] == 0

    # Profile B should have 1 clip
    profile_b = client.get(f"/api/v1/voice-profiles/{profile_b_id}").json()
    assert profile_b["clips"] == 1

    # Recording should point to profile B
    import json as json_mod

    rec_dir = test_settings.storage.base_path / "recordings" / "rec-reassign"
    with open(rec_dir / "metadata.json") as f:
        rec_data = json_mod.load(f)
    assert rec_data["speaker_profiles"]["SPEAKER_00"] == profile_b_id


def test_unassigned_speakers_deleted_profile_treated_as_unassigned(client, test_settings):
    """When speaker_profiles references a deleted profile, speaker is unassigned."""
    # Create and delete a profile
    create_resp = client.post("/api/v1/voice-profiles", json={"name": "Deleted"})
    deleted_id = create_resp.json()["id"]
    client.delete(f"/api/v1/voice-profiles/{deleted_id}")

    # Create recording that references the deleted profile
    _create_recording_in_storage(
        test_settings,
        "rec-deleted-prof",
        speaker_embeddings={"SPEAKER_00": [0.1]},
        speaker_profiles={"SPEAKER_00": deleted_id},
        segments=[
            {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"},
        ],
        speakers=["SPEAKER_00"],
    )

    response = client.get("/api/v1/voice-profiles/unassigned-speakers")
    assert response.status_code == 200

    data = response.json()
    # Should appear as unassigned since the profile was deleted
    assert len(data["speakers"]) == 1
    assert data["speakers"][0]["speaker_label"] == "SPEAKER_00"


def test_unassigned_speakers_no_matching_segments(client, test_settings):
    """Speaker with embedding but no matching segments is skipped."""
    _create_recording_in_storage(
        test_settings,
        "rec-no-segs",
        speaker_embeddings={"SPEAKER_00": [0.1]},
        # Segments reference SPEAKER_01, not SPEAKER_00
        segments=[
            {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_01"},
        ],
        speakers=["SPEAKER_00", "SPEAKER_01"],
    )

    response = client.get("/api/v1/voice-profiles/unassigned-speakers")
    assert response.status_code == 200

    data = response.json()
    # SPEAKER_00 has embedding but no segments → skipped
    # SPEAKER_01 has segments but no embedding → not in speaker_embeddings → skipped
    assert len(data["speakers"]) == 0


def test_unassigned_speakers_processing_recordings_excluded(client, test_settings):
    """Processing recordings should not show unassigned speakers."""
    _create_recording_in_storage(
        test_settings,
        "rec-processing",
        status="processing",
        speaker_embeddings={"SPEAKER_00": [0.1]},
        segments=[
            {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"},
        ],
    )

    response = client.get("/api/v1/voice-profiles/unassigned-speakers")
    data = response.json()
    assert len(data["speakers"]) == 0


def test_profile_clips_deleted_recording_skipped(client, test_settings):
    """Clips endpoint skips recordings that no longer exist."""
    # Create profile and add an embedding manually
    create_resp = client.post("/api/v1/voice-profiles", json={"name": "Orphan"})
    profile_id = create_resp.json()["id"]

    # Create recording, assign, then delete recording
    _create_recording_in_storage(
        test_settings,
        "rec-delete-me",
        speaker_embeddings={"SPEAKER_00": [0.1]},
        segments=[
            {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"},
        ],
    )
    client.post(
        f"/api/v1/voice-profiles/{profile_id}/clips",
        json={"recording_id": "rec-delete-me", "speaker_label": "SPEAKER_00"},
    )

    # Delete the recording
    client.delete("/api/v1/recordings/rec-delete-me")

    # Clips endpoint should not crash, should return empty clips
    response = client.get(f"/api/v1/voice-profiles/{profile_id}/clips")
    assert response.status_code == 200
    data = response.json()
    assert len(data["clips"]) == 0


def test_assign_clip_profile_not_found(client, test_settings):
    """Test 404 when assigning to nonexistent profile."""
    _create_recording_in_storage(
        test_settings,
        "rec-noprof",
        speaker_embeddings={"SPEAKER_00": [0.1]},
        segments=[
            {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"},
        ],
    )

    response = client.post(
        "/api/v1/voice-profiles/nonexistent-profile/clips",
        json={"recording_id": "rec-noprof", "speaker_label": "SPEAKER_00"},
    )
    assert response.status_code == 404


def test_remove_clip_profile_not_found(client):
    """Test 404 when removing clip from nonexistent profile."""
    response = client.delete("/api/v1/voice-profiles/nonexistent-profile/clips/rec-1/SPEAKER_00")
    assert response.status_code == 404


def test_assign_clip_updates_total_seconds(client, test_settings):
    """Test that assigning a clip correctly calculates total_seconds from segments."""
    create_resp = client.post("/api/v1/voice-profiles", json={"name": "Duration"})
    profile_id = create_resp.json()["id"]

    # Recording with 3 segments for SPEAKER_00 totaling 15 seconds
    _create_recording_in_storage(
        test_settings,
        "rec-duration",
        speaker_embeddings={"SPEAKER_00": [0.1]},
        segments=[
            {"start": 0.0, "end": 5.0, "text": "Seg1", "speaker": "SPEAKER_00"},
            {"start": 10.0, "end": 15.0, "text": "Seg2", "speaker": "SPEAKER_00"},
            {"start": 20.0, "end": 25.0, "text": "Seg3", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "text": "Other", "speaker": "SPEAKER_01"},
        ],
    )

    response = client.post(
        f"/api/v1/voice-profiles/{profile_id}/clips",
        json={"recording_id": "rec-duration", "speaker_label": "SPEAKER_00"},
    )
    assert response.status_code == 201

    profile = client.get(f"/api/v1/voice-profiles/{profile_id}").json()
    assert profile["total_seconds"] == 15.0  # 3 x 5s segments


def test_unassigned_speakers_segments_sorted_by_duration_desc(client, test_settings):
    """Test that segments are sorted longest first."""
    _create_recording_in_storage(
        test_settings,
        "rec-sorted",
        speaker_embeddings={"SPEAKER_00": [0.1]},
        segments=[
            {"start": 0.0, "end": 1.0, "text": "Short", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 11.0, "text": "Long", "speaker": "SPEAKER_00"},
            {"start": 11.0, "end": 14.0, "text": "Medium", "speaker": "SPEAKER_00"},
        ],
    )

    response = client.get("/api/v1/voice-profiles/unassigned-speakers")
    data = response.json()

    assert len(data["speakers"]) == 1
    segs = data["speakers"][0]["segments"]
    # Should be sorted by duration desc: 10s, 3s, 1s
    durations = [s["end"] - s["start"] for s in segs]
    assert durations == [10.0, 3.0, 1.0]


def test_unassigned_speakers_max_20_segments(client, test_settings):
    """Test that only top 20 segments are returned per speaker."""
    # Create 25 segments
    segments = [
        {"start": float(i), "end": float(i + 1), "text": f"Seg {i}", "speaker": "SPEAKER_00"}
        for i in range(25)
    ]

    _create_recording_in_storage(
        test_settings,
        "rec-many-segs",
        speaker_embeddings={"SPEAKER_00": [0.1]},
        segments=segments,
    )

    response = client.get("/api/v1/voice-profiles/unassigned-speakers")
    data = response.json()

    assert len(data["speakers"]) == 1
    assert len(data["speakers"][0]["segments"]) == 20


def test_unassigned_speakers_diarized_none_with_embeddings(client, test_settings):
    """Recordings with diarized=None but with embeddings should still show speakers.

    Pre-diarized-field recordings may have speaker_embeddings but diarized=None.
    """
    _create_recording_in_storage(
        test_settings,
        "rec-legacy",
        diarized=None,
        speaker_embeddings={"SPEAKER_00": [0.1]},
        segments=[
            {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"},
        ],
        speakers=["SPEAKER_00"],
    )

    response = client.get("/api/v1/voice-profiles/unassigned-speakers")
    assert response.status_code == 200

    data = response.json()
    assert len(data["speakers"]) == 1
    assert data["speakers"][0]["speaker_label"] == "SPEAKER_00"
