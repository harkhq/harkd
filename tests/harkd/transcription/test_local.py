"""Tests for LocalBackend."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from harkd.transcription.backend import TranscriptionRequest
from harkd.transcription.local import LocalBackend

_JSON_DELIMITER = "---HARKD_JSON_RESULT---"


def _make_request(**overrides) -> TranscriptionRequest:
    """Create a test TranscriptionRequest."""
    defaults = {
        "audio_path": Path("/tmp/test.wav"),
        "model_name": "base",
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


class TestLocalBackend:
    """Tests for LocalBackend."""

    @pytest.mark.asyncio
    async def test_transcribe_success(self):
        """Test successful transcription via subprocess."""
        backend = LocalBackend()
        request = _make_request()

        result_json = (
            '{"text":"Hello","language":"en","language_probability":0.9,'
            '"duration":1.0,"segments":[]}'
        )
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = f"{_JSON_DELIMITER}\n{result_json}"
        mock_proc.stderr = ""

        with patch("harkd.transcription.local.subprocess.run", return_value=mock_proc):
            result = await backend.transcribe(request)

        assert result["text"] == "Hello"
        assert result["language"] == "en"

    @pytest.mark.asyncio
    async def test_transcribe_timeout(self):
        """Test subprocess timeout raises RuntimeError."""
        backend = LocalBackend()
        request = _make_request(timeout=10)

        with patch(
            "harkd.transcription.local.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="test", timeout=10),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                await backend.transcribe(request)

    @pytest.mark.asyncio
    async def test_transcribe_nonzero_exit(self):
        """Test subprocess failure raises RuntimeError."""
        backend = LocalBackend()
        request = _make_request()

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "Model not found"

        with patch("harkd.transcription.local.subprocess.run", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="Transcription failed"):
                await backend.transcribe(request)

    @pytest.mark.asyncio
    async def test_transcribe_invalid_json(self):
        """Test invalid JSON output raises error."""
        backend = LocalBackend()
        request = _make_request()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "not json at all"
        mock_proc.stderr = ""

        with patch("harkd.transcription.local.subprocess.run", return_value=mock_proc):
            with pytest.raises(Exception) as exc_info:
                await backend.transcribe(request)
            assert isinstance(exc_info.value, (json.JSONDecodeError, IndexError))

    @pytest.mark.asyncio
    async def test_health_check_always_healthy(self):
        """Test local backend health check always returns healthy."""
        backend = LocalBackend()
        health = await backend.health_check()
        assert health.healthy is True
        assert health.details["backend"] == "local"

    @pytest.mark.asyncio
    async def test_subprocess_command_includes_diarization_params(self):
        """Test that diarization params are passed to subprocess."""
        backend = LocalBackend()
        request = _make_request(
            diarize=True,
            hf_token="hf_test",
            num_speakers=3,
            min_speakers=2,
            max_speakers=5,
            clustering_threshold=0.7,
        )

        result_json = (
            '{"text":"","language":"en","language_probability":0.9,"duration":1.0,"segments":[]}'
        )
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = f"{_JSON_DELIMITER}\n{result_json}"
        mock_proc.stderr = ""

        with patch("harkd.transcription.local.subprocess.run", return_value=mock_proc) as mock_run:
            await backend.transcribe(request)

        cmd = mock_run.call_args[0][0]
        assert "true" in cmd  # diarize=True
        assert "hf_test" in cmd
        assert "3" in cmd  # num_speakers
        assert "2" in cmd  # min_speakers
        assert "5" in cmd  # max_speakers
        assert "0.7" in cmd  # clustering_threshold
