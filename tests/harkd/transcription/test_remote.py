"""Tests for RemoteBackend."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from harkd.exceptions import RemoteTranscriptionError
from harkd.transcription.backend import TranscriptionRequest
from harkd.transcription.remote import RemoteBackend


def _make_request(**overrides) -> TranscriptionRequest:
    """Create a test TranscriptionRequest."""
    defaults = {
        "audio_path": Path("/tmp/test.wav"),
        "model_name": "large-v3",
        "language": None,
        "word_timestamps": False,
        "diarize": False,
        "hf_token": None,
        "beam_size": 3,
        "batch_size": 16,
        "vad_onset": 0.5,
        "vad_offset": 0.363,
        "vad_method": "pyannote",
        "timeout": 1800,
    }
    defaults.update(overrides)
    return TranscriptionRequest(**defaults)


class TestRemoteBackend:
    """Tests for RemoteBackend base class."""

    def test_init_strips_trailing_slash(self):
        """Test endpoint URL has trailing slash stripped."""
        backend = RemoteBackend("https://example.com/api/", worker_api_key="test")
        assert backend._endpoint == "https://example.com/api"

    def test_build_headers_with_worker_key(self):
        """Test headers include worker API key."""
        backend = RemoteBackend("https://example.com", worker_api_key="my-secret")
        headers = backend._build_headers()
        assert headers["Authorization"] == "Bearer my-secret"

    def test_build_headers_without_worker_key(self):
        """Test headers without worker API key."""
        backend = RemoteBackend("https://example.com")
        headers = backend._build_headers()
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_transcribe_success(self, tmp_path):
        """Test successful remote transcription."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")
        request = _make_request(audio_path=audio_file)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "Hello",
            "language": "en",
            "language_probability": 0.95,
            "duration": 1.0,
            "segments": [],
        }

        backend = RemoteBackend("https://example.com", worker_api_key="key")
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        result = await backend.transcribe(request)
        assert result["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_transcribe_4xx_raises_immediately(self, tmp_path):
        """Test 4xx errors raise immediately without retry."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")
        request = _make_request(audio_path=audio_file)

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        backend = RemoteBackend("https://example.com", max_retries=2)
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(RemoteTranscriptionError, match="HTTP 401"):
            await backend.transcribe(request)

        # Should only be called once (no retries for 4xx)
        assert backend._client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_transcribe_5xx_retries(self, tmp_path):
        """Test 5xx errors trigger retries."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")
        request = _make_request(audio_path=audio_file)

        error_response = MagicMock()
        error_response.status_code = 500
        error_response.text = "Internal Server Error"

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"text": "ok", "segments": []}

        backend = RemoteBackend("https://example.com", max_retries=2)
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(side_effect=[error_response, success_response])

        result = await backend.transcribe(request)
        assert result["text"] == "ok"
        assert backend._client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_transcribe_timeout_retries(self, tmp_path):
        """Test timeout errors trigger retries."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")
        request = _make_request(audio_path=audio_file)

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"text": "ok", "segments": []}

        backend = RemoteBackend("https://example.com", max_retries=2)
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(
            side_effect=[httpx.TimeoutException("timeout"), success_response]
        )

        result = await backend.transcribe(request)
        assert result["text"] == "ok"

    @pytest.mark.asyncio
    async def test_transcribe_exhausts_retries(self, tmp_path):
        """Test exhausted retries raise RemoteTranscriptionError."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")
        request = _make_request(audio_path=audio_file)

        backend = RemoteBackend("https://example.com", max_retries=1)
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        with pytest.raises(RemoteTranscriptionError, match="after 2 attempts"):
            await backend.transcribe(request)

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test healthy health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "model_loaded": True}

        backend = RemoteBackend("https://example.com")
        backend._client = AsyncMock()
        backend._client.get = AsyncMock(return_value=mock_response)

        health = await backend.health_check()
        assert health.healthy is True
        assert health.details["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test unhealthy health check."""
        backend = RemoteBackend("https://example.com")
        backend._client = AsyncMock()
        backend._client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        health = await backend.health_check()
        assert health.healthy is False

    @pytest.mark.asyncio
    async def test_transcribe_5xx_exhausts_retries_gives_consistent_error(self, tmp_path):
        """Test 5xx on ALL attempts gives 'after N attempts' error like timeouts do."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")
        request = _make_request(audio_path=audio_file)

        error_response = MagicMock()
        error_response.status_code = 500
        error_response.text = "Internal Server Error"

        backend = RemoteBackend("https://example.com", max_retries=1)
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=error_response)

        with pytest.raises(RemoteTranscriptionError, match="after 2 attempts"):
            await backend.transcribe(request)

        # 1 initial + 1 retry = 2 total attempts
        assert backend._client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_transcribe_5xx_on_last_attempt_same_as_timeout(self, tmp_path):
        """Test that 5xx on last attempt gives same error format as timeout."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")
        request = _make_request(audio_path=audio_file)

        error_response = MagicMock()
        error_response.status_code = 503
        error_response.text = "Service Unavailable"

        # max_retries=0 means only 1 attempt, no retries
        backend = RemoteBackend("https://example.com", max_retries=0)
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=error_response)

        with pytest.raises(RemoteTranscriptionError, match="after 1 attempts"):
            await backend.transcribe(request)

        assert backend._client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_transcribe_connect_error_retries(self, tmp_path):
        """Test connection errors trigger retries."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")
        request = _make_request(audio_path=audio_file)

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"text": "ok", "segments": []}

        backend = RemoteBackend("https://example.com", max_retries=2)
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(
            side_effect=[httpx.ConnectError("refused"), success_response]
        )

        result = await backend.transcribe(request)
        assert result["text"] == "ok"
        assert backend._client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing the backend."""
        backend = RemoteBackend("https://example.com")
        backend._client = AsyncMock()
        await backend.close()
        backend._client.aclose.assert_called_once()
