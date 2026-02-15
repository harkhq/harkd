"""Tests for settings API routes (read-only)."""

import pytest
from fastapi.testclient import TestClient

from harkd.api.app import create_app
from harkd.config import HarkdSettings, RecordingDefaults, StorageSettings


@pytest.fixture
def test_settings(tmp_path):
    """Create test settings with defaults."""
    return HarkdSettings(storage=StorageSettings(base_path=tmp_path))


@pytest.fixture
def custom_settings(tmp_path):
    """Create test settings with custom recording defaults."""
    return HarkdSettings(
        storage=StorageSettings(base_path=tmp_path),
        recording=RecordingDefaults(
            model="small",
            language="en",
            diarization=False,
            word_timestamps=True,
        ),
    )


@pytest.fixture
def client(test_settings):
    """Create test client with default settings."""
    from harkd.config import get_settings

    app = create_app(test_settings)
    app.dependency_overrides[get_settings] = lambda: test_settings
    return TestClient(app)


@pytest.fixture
def custom_client(custom_settings):
    """Create test client with custom recording settings."""
    from harkd.config import get_settings

    app = create_app(custom_settings)
    app.dependency_overrides[get_settings] = lambda: custom_settings
    return TestClient(app)


def test_get_settings_returns_daemon_defaults(client):
    """Test GET /settings returns daemon recording defaults."""
    response = client.get("/api/v1/settings")
    assert response.status_code == 200

    data = response.json()
    assert data["model"] == "large-v3"
    assert data["language"] == "auto"
    assert data["diarization"] is True
    assert data["noise_reduction"] is True
    assert data["normalization"] is True
    assert data["word_timestamps"] is False


def test_get_settings_reflects_custom_config(custom_client):
    """Test GET /settings reflects custom daemon configuration."""
    response = custom_client.get("/api/v1/settings")
    assert response.status_code == 200

    data = response.json()
    assert data["model"] == "small"
    assert data["language"] == "en"
    assert data["diarization"] is False
    assert data["word_timestamps"] is True
    # Unchanged defaults
    assert data["noise_reduction"] is True
    assert data["normalization"] is True


def test_patch_settings_not_allowed(client):
    """Test PATCH /settings is no longer available."""
    response = client.patch("/api/v1/settings", json={"model": "small"})
    assert response.status_code == 405


def test_get_settings_no_computed_fields(client):
    """Test GET /settings does not include old computed fields (device, ram_usage)."""
    response = client.get("/api/v1/settings")
    assert response.status_code == 200

    data = response.json()
    assert "device" not in data
    assert "ram_usage" not in data


def test_get_settings_response_model(client):
    """Test response contains all expected recording setting fields."""
    response = client.get("/api/v1/settings")
    assert response.status_code == 200

    data = response.json()
    expected_fields = [
        "model",
        "word_timestamps",
        "language",
        "diarization",
        "noise_reduction",
        "normalization",
    ]
    for field in expected_fields:
        assert field in data, f"Missing field: {field}"
