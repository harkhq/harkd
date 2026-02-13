"""Tests for voice profile API routes."""

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
