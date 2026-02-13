"""Tests for transcription worker."""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _skip_torch_patch(monkeypatch):
    """Prevent the torch serialization patch from running during tests.

    The worker module calls _patch_torch_serialization() at import time,
    which conflicts with an already-loaded torch in the test process.
    We mock torch.serialization to prevent the RuntimeError.
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "torch.serialization", mock_torch.serialization)


def _make_mock_transcriber(text="Hello world", language="en", prob=0.95, dur=2.0):
    """Helper: create a mock Transcriber + result."""
    mock_word = MagicMock(start=0.0, end=1.0, word="Hello")
    mock_segment = MagicMock(start=0.0, end=dur, text=text, words=[mock_word])

    mock_result = MagicMock()
    mock_result.text = text
    mock_result.language = language
    mock_result.language_probability = prob
    mock_result.duration = dur
    mock_result.segments = [mock_segment]

    mock_transcriber = MagicMock()
    mock_transcriber.__enter__ = MagicMock(return_value=mock_transcriber)
    mock_transcriber.__exit__ = MagicMock(return_value=False)
    mock_transcriber.transcribe.return_value = mock_result
    return mock_transcriber


def _make_mock_diarizer():
    """Helper: create a mock Diarizer + result."""
    mock_word = MagicMock(start=0.0, end=1.0, word="Hello", speaker="SPEAKER_01")
    mock_segment = MagicMock(
        start=0.0,
        end=2.0,
        text="Hello world",
        speaker="SPEAKER_01",
        words=[mock_word],
    )
    mock_result = MagicMock()
    mock_result.segments = [mock_segment]
    mock_result.speakers = ["SPEAKER_01"]
    mock_result.language = "en"
    mock_result.language_probability = 0.0
    mock_result.duration = 2.0

    mock_diarizer = MagicMock()
    mock_diarizer.__enter__ = MagicMock(return_value=mock_diarizer)
    mock_diarizer.__exit__ = MagicMock(return_value=False)
    mock_diarizer.transcribe_and_diarize.return_value = mock_result
    return mock_diarizer


class TestTranscribeAudioWorkerTranscriberPath:
    """Tests for transcribe_audio_worker with Transcriber (diarize=False)."""

    def test_returns_correct_structure(self):
        """Test non-diarized path marshals Transcriber output to dict."""
        from harkd.services.transcription_worker import transcribe_audio_worker

        mt = _make_mock_transcriber()
        with patch("harkd.audio.transcriber.Transcriber", return_value=mt):
            result = transcribe_audio_worker("/tmp/audio.wav", "base", None, True, diarize=False)

        assert result["text"] == "Hello world"
        assert result["language"] == "en"
        assert result["language_probability"] == 0.95
        assert result["duration"] == 2.0
        segments = result["segments"]
        assert isinstance(segments, list)
        assert len(segments) == 1
        seg0 = segments[0]
        assert isinstance(seg0, dict)
        assert seg0["text"] == "Hello world"
        words = seg0["words"]
        assert isinstance(words, list)
        assert words[0]["word"] == "Hello"
        # Non-diarized output must NOT have speaker fields
        assert "speaker" not in seg0
        assert "speakers" not in result

    def test_passes_params_to_transcriber(self):
        """Test that model_name, language, word_timestamps reach Transcriber."""
        from harkd.services.transcription_worker import transcribe_audio_worker

        mt = _make_mock_transcriber()
        with patch("harkd.audio.transcriber.Transcriber", return_value=mt) as mock_cls:
            transcribe_audio_worker("/tmp/audio.wav", "large-v3", "de", True)

        mock_cls.assert_called_once_with(
            model_name="large-v3",
            device="auto",
            language="de",
            compute_type="auto",
        )
        mt.transcribe.assert_called_once()
        call_kwargs = mt.transcribe.call_args
        assert call_kwargs[1]["word_timestamps"] is True


class TestTranscribeAudioWorkerDiarizerPath:
    """Tests for transcribe_audio_worker with Diarizer (diarize=True)."""

    def test_diarized_output_includes_speakers(self):
        """Test diarized path marshals Diarizer output with speaker info."""
        from harkd.services.transcription_worker import transcribe_audio_worker

        md = _make_mock_diarizer()
        with patch("harkd.audio.diarizer.Diarizer", return_value=md):
            result = transcribe_audio_worker(
                "/tmp/audio.wav",
                "large-v3",
                None,
                True,
                diarize=True,
                hf_token="hf_test123",
            )

        assert result["speakers"] == ["SPEAKER_01"]
        segments = result["segments"]
        assert isinstance(segments, list)
        seg0 = segments[0]
        assert isinstance(seg0, dict)
        assert seg0["speaker"] == "SPEAKER_01"
        words = seg0["words"]
        assert isinstance(words, list)
        assert words[0]["speaker"] == "SPEAKER_01"
        assert result["text"] == "Hello world"

    def test_passes_params_to_diarizer(self):
        """Test that model_name, hf_token, language reach Diarizer."""
        from harkd.services.transcription_worker import transcribe_audio_worker

        md = _make_mock_diarizer()
        with patch("harkd.audio.diarizer.Diarizer", return_value=md) as mock_cls:
            transcribe_audio_worker(
                "/tmp/audio.wav",
                "large-v3",
                "fr",
                True,
                diarize=True,
                hf_token="hf_mytoken",
            )

        mock_cls.assert_called_once_with(
            model_name="large-v3",
            device="auto",
            hf_token="hf_mytoken",
            compute_type="auto",
        )
        md.transcribe_and_diarize.assert_called_once()
        call_kwargs = md.transcribe_and_diarize.call_args
        assert call_kwargs[1]["language"] == "fr"

    def test_diarize_true_but_no_token_falls_back_to_transcriber(self):
        """Test diarize=True with hf_token=None falls to Transcriber path."""
        from harkd.services.transcription_worker import transcribe_audio_worker

        mt = _make_mock_transcriber(text="Fallback")
        with patch("harkd.audio.transcriber.Transcriber", return_value=mt) as mock_cls:
            result = transcribe_audio_worker(
                "/tmp/audio.wav",
                "base",
                None,
                False,
                diarize=True,
                hf_token=None,
            )

        # Transcriber was used, not Diarizer
        mock_cls.assert_called_once()
        assert result["text"] == "Fallback"
        assert "speakers" not in result


class TestMainBlockCLI:
    """Tests for the __main__ CLI by running the actual module as a subprocess."""

    def test_wrong_arg_count_exits_with_error(self):
        """Test that providing wrong number of args exits with code 1."""
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "harkd.services.transcription_worker",
                "path",
                "model",
                "None",
                "true",
            ],  # 4 args, needs 6
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode == 1
        assert "Usage:" in proc.stdout or "Usage:" in proc.stderr

    def test_correct_args_invokes_worker(self):
        """Test that 6 args reaches the worker function (will fail on actual
        transcription since no audio file, but proves arg parsing works)."""
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "harkd.services.transcription_worker",
                "/nonexistent/audio.wav",
                "base",
                "None",
                "false",
                "false",
                "None",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # The process should fail because the audio file doesn't exist or
        # because Transcriber can't load, but it should NOT fail on arg parsing
        assert proc.returncode != 0
        # Shouldn't see "Usage:" since arg count is correct
        assert "Usage:" not in proc.stdout

    def test_json_delimiter_matches_service(self):
        """Test that the JSON delimiter constant matches recording_service."""
        from harkd.services.recording_service import (
            _JSON_DELIMITER as svc_delim,
        )
        from harkd.services.transcription_worker import _JSON_DELIMITER

        assert svc_delim == _JSON_DELIMITER
