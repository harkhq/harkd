"""Tests for VerdaBackend."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harkd.config import VerdaProviderSettings
from harkd.exceptions import RemoteTranscriptionError
from harkd.transcription.backend import TranscriptionRequest
from harkd.transcription.verda import VerdaBackend


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


class TestVerdaBackend:
    """Tests for Verda async polling behavior."""

    def test_auth_headers(self):
        """Test Verda auth headers."""
        backend = VerdaBackend(
            endpoint_url="https://containers.datacrunch.io/hark",
            provider=VerdaProviderSettings(api_key="dc_inf_test"),
        )
        headers = backend._build_headers()
        assert headers["Authorization"] == "Bearer dc_inf_test"

    def test_auth_headers_with_worker_key(self):
        """Test headers include worker API key in separate header."""
        backend = VerdaBackend(
            endpoint_url="https://containers.datacrunch.io/hark",
            provider=VerdaProviderSettings(api_key="dc_inf_test"),
            worker_api_key="worker-secret",
        )
        headers = backend._build_headers()
        assert headers["Authorization"] == "Bearer dc_inf_test"
        assert headers["X-Worker-Api-Key"] == "worker-secret"

    @pytest.mark.asyncio
    async def test_poll_until_complete_success(self):
        """Test polling completes when status=3."""
        backend = VerdaBackend(
            endpoint_url="https://containers.datacrunch.io/hark",
            provider=VerdaProviderSettings(api_key="dc_inf_test", poll_interval=1),
        )

        # Status response: processing then complete
        status_processing = MagicMock()
        status_processing.status_code = 200
        status_processing.json.return_value = {"Status": 2}

        status_complete = MagicMock()
        status_complete.status_code = 200
        status_complete.json.return_value = {"Status": 3}

        # Result response
        result_response = MagicMock()
        result_response.status_code = 200
        result_response.json.return_value = {
            "text": "Hello",
            "language": "en",
            "segments": [],
        }

        backend._client = AsyncMock()
        backend._client.get = AsyncMock(
            side_effect=[status_processing, status_complete, result_response]
        )

        task_info = {
            "Id": "task-123",
            "StatusPath": "/status/deploy1",
            "ResultPath": "/result/deploy1",
        }

        with patch("harkd.transcription.verda.asyncio.sleep", new_callable=AsyncMock):
            result = await backend._poll_until_complete(task_info)

        assert result["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_poll_missing_paths_raises(self):
        """Test missing StatusPath/ResultPath raises error."""
        backend = VerdaBackend(
            endpoint_url="https://containers.datacrunch.io/hark",
            provider=VerdaProviderSettings(api_key="dc_inf_test"),
        )

        with pytest.raises(RemoteTranscriptionError, match="missing StatusPath"):
            await backend._poll_until_complete({"Id": "task-123"})

    @pytest.mark.asyncio
    async def test_poll_unknown_status_raises(self):
        """Test unknown status code raises error."""
        backend = VerdaBackend(
            endpoint_url="https://containers.datacrunch.io/hark",
            provider=VerdaProviderSettings(api_key="dc_inf_test", poll_interval=1),
        )

        bad_status = MagicMock()
        bad_status.status_code = 200
        bad_status.json.return_value = {"Status": 99}

        backend._client = AsyncMock()
        backend._client.get = AsyncMock(return_value=bad_status)

        task_info = {
            "Id": "task-123",
            "StatusPath": "/status/deploy1",
            "ResultPath": "/result/deploy1",
        }

        with patch("harkd.transcription.verda.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RemoteTranscriptionError, match="status 99"):
                await backend._poll_until_complete(task_info)

    @pytest.mark.asyncio
    async def test_transcribe_via_upload(self, tmp_path):
        """Test direct upload path (no S3)."""
        backend = VerdaBackend(
            endpoint_url="https://containers.datacrunch.io/hark",
            provider=VerdaProviderSettings(api_key="dc_inf_test", poll_interval=1),
        )

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"audio data")
        request = _make_request(audio_path=audio_file)

        # Submit response
        submit_response = MagicMock()
        submit_response.status_code = 200
        submit_response.json.return_value = {
            "Id": "task-123",
            "StatusPath": "/status/deploy1",
            "ResultPath": "/result/deploy1",
        }

        # Status complete
        status_complete = MagicMock()
        status_complete.status_code = 200
        status_complete.json.return_value = {"Status": 3}

        # Result
        result_response = MagicMock()
        result_response.status_code = 200
        result_response.json.return_value = {"text": "ok", "segments": []}

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=submit_response)
        backend._client.get = AsyncMock(side_effect=[status_complete, result_response])

        with patch("harkd.transcription.verda.asyncio.sleep", new_callable=AsyncMock):
            result = await backend.transcribe(request)

        assert result["text"] == "ok"

    @pytest.mark.asyncio
    async def test_poll_timeout_raises_error(self):
        """Test polling times out after max_poll_seconds instead of polling forever."""
        # Use very short timeout (10s) and poll_interval (5s) so it times out after 2 polls
        backend = VerdaBackend(
            endpoint_url="https://containers.datacrunch.io/hark",
            provider=VerdaProviderSettings(api_key="dc_inf_test", poll_interval=5),
            timeout=10,
        )

        # Always return "processing" status — never completes
        status_processing = MagicMock()
        status_processing.status_code = 200
        status_processing.json.return_value = {"Status": 2}

        backend._client = AsyncMock()
        backend._client.get = AsyncMock(return_value=status_processing)

        task_info = {
            "Id": "task-stuck",
            "StatusPath": "/status/deploy1",
            "ResultPath": "/result/deploy1",
        }

        with patch("harkd.transcription.verda.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RemoteTranscriptionError, match="timed out"):
                await backend._poll_until_complete(task_info)

    @pytest.mark.asyncio
    async def test_poll_status_error_raises_immediately(self):
        """Test HTTP error during polling raises immediately."""
        backend = VerdaBackend(
            endpoint_url="https://containers.datacrunch.io/hark",
            provider=VerdaProviderSettings(api_key="dc_inf_test", poll_interval=1),
        )

        error_response = MagicMock()
        error_response.status_code = 500
        error_response.text = "Internal Server Error"

        backend._client = AsyncMock()
        backend._client.get = AsyncMock(return_value=error_response)

        task_info = {
            "Id": "task-123",
            "StatusPath": "/status/deploy1",
            "ResultPath": "/result/deploy1",
        }

        with patch("harkd.transcription.verda.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RemoteTranscriptionError, match="status poll failed"):
                await backend._poll_until_complete(task_info)

    @pytest.mark.asyncio
    async def test_submit_failure_raises(self, tmp_path):
        """Test job submission failure raises error."""
        backend = VerdaBackend(
            endpoint_url="https://containers.datacrunch.io/hark",
            provider=VerdaProviderSettings(api_key="dc_inf_test", poll_interval=1),
        )

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"audio data")
        request = _make_request(audio_path=audio_file)

        error_response = MagicMock()
        error_response.status_code = 422
        error_response.text = "Unprocessable Entity"

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=error_response)

        with pytest.raises(RemoteTranscriptionError, match="submission failed"):
            await backend.transcribe(request)

    @pytest.mark.asyncio
    async def test_s3_upload_cleanup_on_poll_error(self):
        """Test S3 object is cleaned up even when polling fails."""
        mock_file_storage = AsyncMock()
        mock_file_storage.upload = AsyncMock(return_value=("https://s3/presigned", "key/audio.wav"))
        mock_file_storage.delete = AsyncMock()

        backend = VerdaBackend(
            endpoint_url="https://containers.datacrunch.io/hark",
            provider=VerdaProviderSettings(api_key="dc_inf_test", poll_interval=1),
        )
        backend._file_storage = mock_file_storage

        request = _make_request()

        # Submission succeeds but status check fails
        submit_response = MagicMock()
        submit_response.status_code = 200
        submit_response.json.return_value = {
            "Id": "task-123",
            "StatusPath": "/status/deploy1",
            "ResultPath": "/result/deploy1",
        }

        error_status = MagicMock()
        error_status.status_code = 500
        error_status.text = "Error"

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=submit_response)
        backend._client.get = AsyncMock(return_value=error_status)

        with patch("harkd.transcription.verda.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RemoteTranscriptionError):
                await backend.transcribe(request)

        # S3 cleanup should still happen despite the error
        mock_file_storage.delete.assert_called_once_with("key/audio.wav")
