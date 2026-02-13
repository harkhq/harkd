"""Tests for transcriber."""

import sys
from unittest.mock import MagicMock

import pytest

from harkd.audio.transcriber import (
    Transcriber,
    TranscriptionResult,
    TranscriptionSegment,
    WordSegment,
)


@pytest.fixture
def temp_audio(tmp_path):
    """Create temporary audio file path."""
    audio_file = tmp_path / "test.wav"
    # Create an empty file so exists() check passes
    audio_file.touch()
    return audio_file


def test_transcriber_initialization():
    """Test transcriber can be initialized."""
    transcriber = Transcriber(model_name="base", device="cpu")

    assert transcriber.model_name == "base"
    assert transcriber.device == "cpu"
    assert transcriber._model is None  # Lazy loading


def test_transcriber_with_language():
    """Test transcriber with specific language."""
    transcriber = Transcriber(model_name="base", language="en")

    assert transcriber.language == "en"


def test_transcriber_lazy_loads_model(mock_whisper_model, temp_audio):
    """Test that model is lazy-loaded on first transcribe."""
    transcriber = Transcriber(model_name="base", device="cpu")

    # Model should not be loaded yet
    assert transcriber._model is None
    mock_whisper_model.assert_not_called()

    # Mock transcribe result (WhisperX returns dict)
    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance

    # WhisperX API: transcribe returns dict with "segments" and "language"
    mock_model_instance.transcribe.return_value = {
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Hello world",
                "words": [],
            }
        ],
        "language": "en",
    }

    # Transcribe
    result = transcriber.transcribe(temp_audio)

    # Model should now be loaded
    mock_whisper_model.assert_called_once()
    assert transcriber._model is not None
    assert result.text == "Hello world"


def test_transcriber_transcribe_basic(mock_whisper_model, temp_audio):
    """Test basic transcription."""
    transcriber = Transcriber(model_name="base", device="cpu")

    # Mock model
    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance

    # WhisperX API: transcribe returns dict with "segments" and "language"
    mock_model_instance.transcribe.return_value = {
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "Hello world",
                "words": [],
            },
            {
                "start": 3.0,
                "end": 5.0,
                "text": "How are you",
                "words": [],
            },
        ],
        "language": "en",
    }

    # Transcribe
    result = transcriber.transcribe(temp_audio)

    # Verify result
    assert isinstance(result, TranscriptionResult)
    assert result.text == "Hello world How are you"
    assert len(result.segments) == 2
    assert result.language == "en"
    # WhisperX doesn't expose language probability via its API
    assert result.language_probability == 0.0
    assert result.duration == 5.0

    # Verify segment structure
    seg0 = result.segments[0]
    assert seg0.start == 0.0
    assert seg0.end == 3.0
    assert seg0.text == "Hello world"
    assert seg0.words == []

    seg1 = result.segments[1]
    assert seg1.start == 3.0
    assert seg1.end == 5.0
    assert seg1.text == "How are you"


def test_transcriber_with_word_timestamps(mock_whisper_model, temp_audio, monkeypatch):
    """Test transcription with word-level timestamps."""

    transcriber = Transcriber(model_name="base")

    # Mock model
    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance

    # WhisperX API: initial transcribe returns dict
    mock_model_instance.transcribe.return_value = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world",
            }
        ],
        "language": "en",
    }

    # Mock load_align_model
    mock_align_model = MagicMock()
    mock_metadata = MagicMock()
    whisperx = sys.modules["whisperx"]
    whisperx.load_align_model.return_value = (mock_align_model, mock_metadata)

    # Mock align result (returns aligned segments with words)
    whisperx.align.return_value = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world",
                "words": [
                    {"start": 0.0, "end": 1.0, "word": "Hello"},
                    {"start": 1.0, "end": 2.0, "word": "world"},
                ],
            }
        ],
        "language": "en",
    }

    # Transcribe with word timestamps
    result = transcriber.transcribe(temp_audio, word_timestamps=True)

    # Verify words are included with timing values
    assert len(result.segments) == 1
    assert len(result.segments[0].words) == 2

    word0 = result.segments[0].words[0]
    assert word0.word == "Hello"
    assert word0.start == 0.0
    assert word0.end == 1.0

    word1 = result.segments[0].words[1]
    assert word1.word == "world"
    assert word1.start == 1.0
    assert word1.end == 2.0


def test_transcriber_device_auto_detection(mock_whisper_model, temp_audio):
    """Test auto device detection."""

    transcriber = Transcriber(model_name="base", device="auto")

    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance

    # WhisperX API: transcribe returns dict
    mock_model_instance.transcribe.return_value = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "test",
                "words": [],
            }
        ],
        "language": "en",
    }

    # torch is already mocked in the fixture with is_available returning False
    # Transcribe (will trigger device detection)
    transcriber.transcribe(temp_audio)

    # Should have used CPU (since mock torch.cuda.is_available returns False)
    call_kwargs = mock_whisper_model.call_args[1]
    assert call_kwargs["device"] == "cpu"


def test_transcriber_compute_type_auto(mock_whisper_model, temp_audio):
    """Test auto compute type selection."""
    transcriber = Transcriber(model_name="base", device="cpu", compute_type="auto")

    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance

    # WhisperX API: transcribe returns dict
    mock_model_instance.transcribe.return_value = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "test",
                "words": [],
            }
        ],
        "language": "en",
    }

    # Transcribe
    transcriber.transcribe(temp_audio)

    # Should have used int8 for CPU
    call_kwargs = mock_whisper_model.call_args[1]
    assert call_kwargs["compute_type"] == "int8"


def test_transcription_segment_dataclass():
    """Test TranscriptionSegment dataclass."""
    segment = TranscriptionSegment(start=0.0, end=5.0, text="Test")

    assert segment.start == 0.0
    assert segment.end == 5.0
    assert segment.text == "Test"
    assert segment.words == []


def test_word_segment_dataclass():
    """Test WordSegment dataclass."""
    word = WordSegment(start=0.0, end=1.0, word="Hello")

    assert word.start == 0.0
    assert word.end == 1.0
    assert word.word == "Hello"


class TestTranscriberCloseAndContextManager:
    """Tests for close() and context manager protocol."""

    def test_close_releases_model(self):
        """Test that close() sets _model to None."""
        transcriber = Transcriber(model_name="base")
        transcriber._model = MagicMock()
        transcriber._actual_device = "cpu"

        transcriber.close()

        assert transcriber._model is None
        assert transcriber._actual_device is None  # type: ignore[comparison-overlap]

    def test_context_manager_protocol(self):
        """Test __enter__/__exit__ context manager."""
        transcriber = Transcriber(model_name="base")
        transcriber._model = MagicMock()

        with transcriber as t:
            assert t is transcriber
            assert t._model is not None

        # After exiting context, model should be released
        assert transcriber._model is None
