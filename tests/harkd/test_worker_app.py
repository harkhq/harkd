"""Tests for the GPU worker FastAPI app."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _mock_torch(monkeypatch):
    """Mock torch to avoid GPU dependencies in tests."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.cuda.get_device_name.side_effect = RuntimeError("No CUDA")
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "torch.serialization", mock_torch.serialization)


@pytest.fixture
def _set_api_key(monkeypatch):
    """Set worker API key."""
    monkeypatch.setenv("HARKD_WORKER_API_KEY", "test-secret-key")


@pytest.fixture
def _no_api_key(monkeypatch):
    """Ensure no worker API key is set."""
    monkeypatch.delenv("HARKD_WORKER_API_KEY", raising=False)


class TestWorkerAuth:
    """Tests for worker API key authentication middleware."""

    def test_health_endpoint_is_public(self, _set_api_key):
        """Test /health doesn't require auth."""
        # Re-import to pick up env var
        import importlib

        import harkd.worker.app as app_mod

        importlib.reload(app_mod)
        from fastapi.testclient import TestClient

        client = TestClient(app_mod.app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_transcribe_requires_auth(self, _set_api_key):
        """Test /transcribe rejects requests without auth."""
        import importlib

        import harkd.worker.app as app_mod

        importlib.reload(app_mod)
        from fastapi.testclient import TestClient

        client = TestClient(app_mod.app)
        resp = client.post(
            "/transcribe",
            data={"params": json.dumps({"model_name": "base"})},
        )
        assert resp.status_code == 401

    def test_transcribe_accepts_authorization_header(self, _set_api_key):
        """Test /transcribe accepts Authorization: Bearer <key>."""
        import importlib

        import harkd.worker.app as app_mod

        importlib.reload(app_mod)
        from fastapi.testclient import TestClient

        client = TestClient(app_mod.app)
        # Will fail with 400 (no audio) but auth should pass
        resp = client.post(
            "/transcribe",
            headers={"Authorization": "Bearer test-secret-key"},
            data={"params": json.dumps({"model_name": "base"})},
        )
        # 400 = auth passed, but no audio provided
        assert resp.status_code == 400
        assert "audio" in resp.json()["error"].lower()

    def test_transcribe_accepts_x_worker_api_key_header(self, _set_api_key):
        """Test /transcribe accepts X-Worker-Api-Key header (for Verda)."""
        import importlib

        import harkd.worker.app as app_mod

        importlib.reload(app_mod)
        from fastapi.testclient import TestClient

        client = TestClient(app_mod.app)
        resp = client.post(
            "/transcribe",
            headers={"X-Worker-Api-Key": "test-secret-key"},
            data={"params": json.dumps({"model_name": "base"})},
        )
        # 400 = auth passed, but no audio provided
        assert resp.status_code == 400
        assert "audio" in resp.json()["error"].lower()

    def test_transcribe_rejects_wrong_key(self, _set_api_key):
        """Test /transcribe rejects wrong API key."""
        import importlib

        import harkd.worker.app as app_mod

        importlib.reload(app_mod)
        from fastapi.testclient import TestClient

        client = TestClient(app_mod.app)
        resp = client.post(
            "/transcribe",
            headers={"Authorization": "Bearer wrong-key"},
            data={"params": json.dumps({"model_name": "base"})},
        )
        assert resp.status_code == 401

    def test_no_api_key_configured_rejects_all(self, _no_api_key):
        """Test fail-closed when no API key is configured."""
        import importlib

        import harkd.worker.app as app_mod

        importlib.reload(app_mod)
        from fastapi.testclient import TestClient

        client = TestClient(app_mod.app)
        resp = client.post(
            "/transcribe",
            data={"params": json.dumps({"model_name": "base"})},
        )
        assert resp.status_code == 403
        assert "No API key configured" in resp.json()["error"]


class TestWorkerTranscribe:
    """Tests for /transcribe endpoint behavior."""

    def test_transcribe_with_audio_file(self, _set_api_key, tmp_path):
        """Test /transcribe with multipart file upload."""
        import importlib

        import harkd.worker.app as app_mod

        importlib.reload(app_mod)
        from fastapi.testclient import TestClient

        mock_result = {
            "text": "Hello world",
            "language": "en",
            "language_probability": 0.95,
            "duration": 2.0,
            "segments": [],
        }

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF fake audio data")

        client = TestClient(app_mod.app)
        with patch(
            "harkd.services.transcription_worker.transcribe_audio_worker",
            return_value=mock_result,
        ):
            resp = client.post(
                "/transcribe",
                headers={"Authorization": "Bearer test-secret-key"},
                files={"audio": ("test.wav", audio_file.read_bytes(), "audio/wav")},
                data={"params": json.dumps({"model_name": "base"})},
            )

        assert resp.status_code == 200
        assert resp.json()["text"] == "Hello world"

    def test_transcribe_no_audio_returns_400(self, _set_api_key):
        """Test /transcribe without audio or audio_url returns 400."""
        import importlib

        import harkd.worker.app as app_mod

        importlib.reload(app_mod)
        from fastapi.testclient import TestClient

        client = TestClient(app_mod.app)
        resp = client.post(
            "/transcribe",
            headers={"Authorization": "Bearer test-secret-key"},
            data={"params": json.dumps({"model_name": "base"})},
        )
        assert resp.status_code == 400

    def test_transcribe_passes_all_params(self, _set_api_key, tmp_path):
        """Test that all params from JSON are passed to transcribe_audio_worker."""
        import importlib

        import harkd.worker.app as app_mod

        importlib.reload(app_mod)
        from fastapi.testclient import TestClient

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF fake audio")

        params = {
            "model_name": "large-v3",
            "language": "de",
            "word_timestamps": True,
            "diarize": True,
            "hf_token": "hf_test",
            "beam_size": 5,
            "batch_size": 8,
            "vad_onset": 0.6,
            "vad_offset": 0.4,
            "vad_method": "silero",
            "num_speakers": 3,
            "min_speakers": 2,
            "max_speakers": 5,
            "clustering_threshold": 0.7,
        }

        mock_result = {"text": "", "segments": []}

        client = TestClient(app_mod.app)
        with patch(
            "harkd.services.transcription_worker.transcribe_audio_worker",
            return_value=mock_result,
        ) as mock_worker:
            resp = client.post(
                "/transcribe",
                headers={"Authorization": "Bearer test-secret-key"},
                files={"audio": ("test.wav", audio_file.read_bytes(), "audio/wav")},
                data={"params": json.dumps(params)},
            )

        assert resp.status_code == 200
        call_kwargs = mock_worker.call_args
        assert call_kwargs.kwargs["model_name"] == "large-v3"
        assert call_kwargs.kwargs["language"] == "de"
        assert call_kwargs.kwargs["word_timestamps"] is True
        assert call_kwargs.kwargs["diarize"] is True
        assert call_kwargs.kwargs["hf_token"] == "hf_test"
        assert call_kwargs.kwargs["beam_size"] == 5
        assert call_kwargs.kwargs["num_speakers"] == 3

    def test_transcribe_cleans_up_temp_file(self, _set_api_key, tmp_path):
        """Test temp file is deleted after successful transcription."""
        import importlib

        import harkd.worker.app as app_mod

        importlib.reload(app_mod)
        from fastapi.testclient import TestClient

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF fake audio data")

        created_paths = []
        original_named_temp = __import__("tempfile").NamedTemporaryFile

        def tracking_temp(**kwargs):
            f = original_named_temp(**kwargs)
            created_paths.append(f.name)
            return f

        mock_result = {"text": "ok", "segments": []}
        client = TestClient(app_mod.app)
        with (
            patch("harkd.worker.app.tempfile.NamedTemporaryFile", side_effect=tracking_temp),
            patch(
                "harkd.services.transcription_worker.transcribe_audio_worker",
                return_value=mock_result,
            ),
        ):
            resp = client.post(
                "/transcribe",
                headers={"Authorization": "Bearer test-secret-key"},
                files={"audio": ("test.wav", audio_file.read_bytes(), "audio/wav")},
                data={"params": json.dumps({"model_name": "base"})},
            )

        assert resp.status_code == 200
        assert len(created_paths) == 1
        assert not os.path.exists(created_paths[0])

    def test_no_audio_cleans_up_temp_file(self, _set_api_key):
        """Test temp file is cleaned up even on the 400 'no audio' path."""
        import importlib

        import harkd.worker.app as app_mod

        importlib.reload(app_mod)
        from fastapi.testclient import TestClient

        created_paths = []
        original_named_temp = __import__("tempfile").NamedTemporaryFile

        def tracking_temp(**kwargs):
            f = original_named_temp(**kwargs)
            created_paths.append(f.name)
            return f

        client = TestClient(app_mod.app)
        with patch("harkd.worker.app.tempfile.NamedTemporaryFile", side_effect=tracking_temp):
            resp = client.post(
                "/transcribe",
                headers={"Authorization": "Bearer test-secret-key"},
                data={"params": json.dumps({"model_name": "base"})},
            )

        assert resp.status_code == 400
        assert len(created_paths) == 1
        assert not os.path.exists(created_paths[0])
