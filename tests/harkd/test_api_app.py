"""Tests for FastAPI application."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from harkd import __version__
from harkd.api.app import create_app
from harkd.api.deps import get_recording_service
from harkd.config import CorsSettings, HarkdSettings, ServerSettings, StorageSettings
from harkd.exceptions import NoLoopbackDeviceError, NoMicrophoneError


@pytest.fixture
def test_settings(tmp_path):
    """Create test settings."""
    return HarkdSettings(
        server=ServerSettings(host="127.0.0.1", port=8765),
        cors=CorsSettings(
            enabled=True,
            origins=["http://localhost:5173"],
            allow_credentials=True,
        ),
        storage=StorageSettings(base_path=tmp_path),
    )


@pytest.fixture
def app(test_settings):
    """Create test FastAPI app."""
    return create_app(test_settings)


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


def test_app_creation(app):
    """Test that app is created successfully with routes registered."""
    assert app.title == "harkd"
    assert app.version == __version__

    # Verify routes are registered
    route_paths = [route.path for route in app.routes]
    assert "/api/v1/health" in route_paths
    assert "/api/v1/recordings" in route_paths
    assert "/api/v1/settings" in route_paths
    assert "/api/v1/voice-profiles" in route_paths


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == __version__
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0


def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    # CORS should allow the origin
    assert "access-control-allow-origin" in response.headers


def test_exception_handler_recording_not_found(client):
    """Test that recording not found returns proper JSON error response."""
    response = client.get("/api/v1/recordings/nonexistent-id-123")
    assert response.status_code == 404

    data = response.json()
    error = data["detail"]["error"]
    assert error["code"] == "RECORDING_NOT_FOUND"
    assert "nonexistent-id-123" in error["message"]
    assert error["details"]["recording_id"] == "nonexistent-id-123"


def test_exception_handler_voice_profile_not_found(client):
    """Test that voice profile not found returns proper JSON error response via global handler."""
    response = client.get("/api/v1/voice-profiles/nonexistent-prof-123")
    assert response.status_code == 404

    data = response.json()
    error = data["error"]
    assert error["code"] == "VOICE_PROFILE_NOT_FOUND"
    assert "nonexistent-prof-123" in error["message"]
    assert error["details"]["profile_id"] == "nonexistent-prof-123"


def test_app_with_cors_disabled(tmp_path):
    """Test app creation with CORS disabled."""
    settings = HarkdSettings(
        cors=CorsSettings(enabled=False),
        storage=StorageSettings(base_path=tmp_path),
    )
    app = create_app(settings)
    client = TestClient(app)

    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_endpoint_uptime_increases(client):
    """Test that uptime increases between calls."""
    import time

    response1 = client.get("/api/v1/health")
    uptime1 = response1.json()["uptime_seconds"]

    time.sleep(0.1)

    response2 = client.get("/api/v1/health")
    uptime2 = response2.json()["uptime_seconds"]

    assert uptime2 > uptime1


def test_exception_handler_no_microphone(app):
    """Test that NO_MICROPHONE error returns 422."""
    mock_service = MagicMock()
    mock_service.start_recording = AsyncMock(side_effect=NoMicrophoneError())
    app.dependency_overrides[get_recording_service] = lambda: mock_service

    try:
        client = TestClient(app)
        response = client.post("/api/v1/recordings", json={})
        assert response.status_code == 422

        data = response.json()
        assert data["error"]["code"] == "NO_MICROPHONE"
    finally:
        app.dependency_overrides.pop(get_recording_service, None)


def test_exception_handler_no_loopback_device(app):
    """Test that NO_LOOPBACK_DEVICE error returns 422."""
    mock_service = MagicMock()
    mock_service.start_recording = AsyncMock(side_effect=NoLoopbackDeviceError())
    app.dependency_overrides[get_recording_service] = lambda: mock_service

    try:
        client = TestClient(app)
        response = client.post("/api/v1/recordings", json={})
        assert response.status_code == 422

        data = response.json()
        assert data["error"]["code"] == "NO_LOOPBACK_DEVICE"
    finally:
        app.dependency_overrides.pop(get_recording_service, None)
