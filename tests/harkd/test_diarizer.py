"""Tests for diarizer."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from harkd.audio.diarizer import (
    DiarizationResult,
    DiarizedSegment,
    Diarizer,
    WordSegment,
    _renumber_speaker,
)


@pytest.fixture
def mock_whisperx():
    """Create mock WhisperX module."""
    with patch("harkd.audio.diarizer.whisperx") as mock:
        yield mock


@pytest.fixture
def temp_audio(tmp_path):
    """Create temporary audio file path."""
    audio_file = tmp_path / "test.wav"
    # Create an empty file so exists() check passes
    audio_file.touch()
    return audio_file


def test_diarizer_initialization():
    """Test diarizer can be initialized."""
    diarizer = Diarizer(model_name="base", device="cpu", hf_token="test_token")

    assert diarizer.model_name == "base"
    assert diarizer.device == "cpu"
    assert diarizer.hf_token == "test_token"
    assert diarizer._model is None  # Lazy loading


def test_diarizer_with_num_speakers():
    """Test diarizer with specific number of speakers."""
    diarizer = Diarizer(model_name="base", num_speakers=2)

    assert diarizer.num_speakers == 2


def test_diarizer_requires_hf_token(temp_audio):
    """Test that diarizer raises error without HF token."""
    diarizer = Diarizer(model_name="base", hf_token=None)

    with pytest.raises(RuntimeError) as exc_info:
        diarizer.transcribe_and_diarize(temp_audio)

    assert "HuggingFace token is required" in str(exc_info.value)


def test_diarizer_lazy_loads_model(temp_audio):
    """Test that model is lazy-loaded on first transcribe."""
    with patch.dict(
        "sys.modules", {"whisperx": MagicMock(), "whisperx.diarize": MagicMock()}
    ):
        wx: Any = sys.modules["whisperx"]
        wx_diarize: Any = sys.modules["whisperx.diarize"]

        diarizer = Diarizer(model_name="base", device="cpu", hf_token="test_token")

        # Model should not be loaded yet
        assert diarizer._model is None

        # Mock WhisperX components
        mock_model = MagicMock()
        wx.load_model.return_value = mock_model
        wx.load_audio.return_value = MagicMock()

        # Mock transcribe result
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [],
        }

        # Mock align
        mock_align_model = MagicMock()
        mock_metadata = MagicMock()
        wx.load_align_model.return_value = (mock_align_model, mock_metadata)
        wx.align.return_value = {"segments": []}

        # Mock diarization
        mock_diarize_pipeline = MagicMock()
        mock_diarize_pipeline.return_value = MagicMock()
        wx_diarize.DiarizationPipeline.return_value = mock_diarize_pipeline

        wx.assign_word_speakers.return_value = {"segments": []}

        # Transcribe
        result = diarizer.transcribe_and_diarize(temp_audio)

        # Model should now be loaded
        wx.load_model.assert_called_once()
        assert diarizer._model is not None
        assert isinstance(result.segments, list)
        assert isinstance(result.speakers, list)


def test_diarizer_transcribe_basic(temp_audio):
    """Test basic diarization."""
    with patch.dict(
        "sys.modules", {"whisperx": MagicMock(), "whisperx.diarize": MagicMock()}
    ):
        wx: Any = sys.modules["whisperx"]
        wx_diarize: Any = sys.modules["whisperx.diarize"]

        diarizer = Diarizer(model_name="base", device="cpu", hf_token="test_token")

        # Mock model
        mock_model = MagicMock()
        wx.load_model.return_value = mock_model
        wx.load_audio.return_value = MagicMock()

        # Create mock segments
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 3.0, "text": "Hello world"},
                {"start": 3.0, "end": 5.0, "text": "How are you"},
            ],
        }

        # Mock align
        mock_align_model = MagicMock()
        mock_metadata = MagicMock()
        wx.load_align_model.return_value = (mock_align_model, mock_metadata)
        wx.align.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "text": "Hello world",
                    "speaker": "SPEAKER_00",
                    "words": [],
                },
                {
                    "start": 3.0,
                    "end": 5.0,
                    "text": "How are you",
                    "speaker": "SPEAKER_01",
                    "words": [],
                },
            ]
        }

        # Mock diarization
        mock_diarize_pipeline = MagicMock()
        mock_diarize_pipeline.return_value = MagicMock()
        wx_diarize.DiarizationPipeline.return_value = mock_diarize_pipeline

        wx.assign_word_speakers.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "text": "Hello world",
                    "speaker": "SPEAKER_00",
                    "words": [],
                },
                {
                    "start": 3.0,
                    "end": 5.0,
                    "text": "How are you",
                    "speaker": "SPEAKER_01",
                    "words": [],
                },
            ]
        }

        # Transcribe and diarize
        result = diarizer.transcribe_and_diarize(temp_audio)

        # Verify result
        assert isinstance(result, DiarizationResult)
        assert len(result.segments) == 2
        assert result.language == "en"
        assert result.duration == 5.0
        # Speakers should be renumbered to SPEAKER_01, SPEAKER_02
        assert "SPEAKER_01" in result.speakers
        assert "SPEAKER_02" in result.speakers

        # Verify segment content and speaker assignment
        seg0 = result.segments[0]
        assert seg0.start == 0.0
        assert seg0.end == 3.0
        assert seg0.text == "Hello world"
        assert seg0.speaker == "SPEAKER_01"

        seg1 = result.segments[1]
        assert seg1.start == 3.0
        assert seg1.end == 5.0
        assert seg1.text == "How are you"
        assert seg1.speaker == "SPEAKER_02"


def test_diarizer_with_word_timestamps(temp_audio):
    """Test diarization with word-level timestamps."""
    with patch.dict(
        "sys.modules", {"whisperx": MagicMock(), "whisperx.diarize": MagicMock()}
    ):
        wx: Any = sys.modules["whisperx"]
        wx_diarize: Any = sys.modules["whisperx.diarize"]

        diarizer = Diarizer(model_name="base", hf_token="test_token")

        # Mock model
        mock_model = MagicMock()
        wx.load_model.return_value = mock_model
        wx.load_audio.return_value = MagicMock()

        # Create mock transcription
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 2.0, "text": "Hello world"}],
        }

        # Mock align with words
        mock_align_model = MagicMock()
        mock_metadata = MagicMock()
        wx.load_align_model.return_value = (mock_align_model, mock_metadata)
        wx.align.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello world",
                    "speaker": "SPEAKER_00",
                    "words": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "word": "Hello",
                            "speaker": "SPEAKER_00",
                        },
                        {
                            "start": 1.0,
                            "end": 2.0,
                            "word": "world",
                            "speaker": "SPEAKER_00",
                        },
                    ],
                }
            ]
        }

        # Mock diarization
        mock_diarize_pipeline = MagicMock()
        mock_diarize_pipeline.return_value = MagicMock()
        wx_diarize.DiarizationPipeline.return_value = mock_diarize_pipeline

        wx.assign_word_speakers.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello world",
                    "speaker": "SPEAKER_00",
                    "words": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "word": "Hello",
                            "speaker": "SPEAKER_00",
                        },
                        {
                            "start": 1.0,
                            "end": 2.0,
                            "word": "world",
                            "speaker": "SPEAKER_00",
                        },
                    ],
                }
            ]
        }

        # Transcribe
        result = diarizer.transcribe_and_diarize(temp_audio)

        # Verify words are included with timing values
        assert len(result.segments) == 1
        assert len(result.segments[0].words) == 2

        word0 = result.segments[0].words[0]
        assert word0.word == "Hello"
        assert word0.start == 0.0
        assert word0.end == 1.0
        assert word0.speaker == "SPEAKER_01"

        word1 = result.segments[0].words[1]
        assert word1.word == "world"
        assert word1.start == 1.0
        assert word1.end == 2.0
        assert word1.speaker == "SPEAKER_01"


def test_diarizer_device_auto_detection(temp_audio):
    """Test auto device detection."""
    # Mock torch module with cuda.is_available returning False
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict(
        "sys.modules",
        {
            "whisperx": MagicMock(),
            "whisperx.diarize": MagicMock(),
            "torch": mock_torch,
        },
    ):
        wx: Any = sys.modules["whisperx"]
        wx_diarize: Any = sys.modules["whisperx.diarize"]

        diarizer = Diarizer(model_name="base", device="auto", hf_token="test_token")

        # Mock model
        mock_model = MagicMock()
        wx.load_model.return_value = mock_model
        wx.load_audio.return_value = MagicMock()

        # Mock transcription
        mock_model.transcribe.return_value = {"language": "en", "segments": []}

        # Mock align
        mock_align_model = MagicMock()
        mock_metadata = MagicMock()
        wx.load_align_model.return_value = (mock_align_model, mock_metadata)
        wx.align.return_value = {"segments": []}

        # Mock diarization
        mock_diarize_pipeline = MagicMock()
        mock_diarize_pipeline.return_value = MagicMock()
        wx_diarize.DiarizationPipeline.return_value = mock_diarize_pipeline

        wx.assign_word_speakers.return_value = {"segments": []}

        # Transcribe (will trigger device detection)
        diarizer.transcribe_and_diarize(temp_audio)

        # Should have used CPU (since we mocked is_available to return False)
        call_kwargs = wx.load_model.call_args[1]
        assert call_kwargs["device"] == "cpu"


def test_diarizer_compute_type_auto(temp_audio):
    """Test auto compute type selection."""
    with patch.dict(
        "sys.modules", {"whisperx": MagicMock(), "whisperx.diarize": MagicMock()}
    ):
        wx: Any = sys.modules["whisperx"]
        wx_diarize: Any = sys.modules["whisperx.diarize"]

        diarizer = Diarizer(
            model_name="base", device="cpu", compute_type="auto", hf_token="test_token"
        )

        # Mock model
        mock_model = MagicMock()
        wx.load_model.return_value = mock_model
        wx.load_audio.return_value = MagicMock()

        # Mock transcription
        mock_model.transcribe.return_value = {"language": "en", "segments": []}

        # Mock align
        mock_align_model = MagicMock()
        mock_metadata = MagicMock()
        wx.load_align_model.return_value = (mock_align_model, mock_metadata)
        wx.align.return_value = {"segments": []}

        # Mock diarization
        mock_diarize_pipeline = MagicMock()
        mock_diarize_pipeline.return_value = MagicMock()
        wx_diarize.DiarizationPipeline.return_value = mock_diarize_pipeline

        wx.assign_word_speakers.return_value = {"segments": []}

        # Transcribe
        diarizer.transcribe_and_diarize(temp_audio)

        # Should have used int8 for CPU
        call_kwargs = wx.load_model.call_args[1]
        assert call_kwargs["compute_type"] == "int8"


def test_diarized_segment_dataclass():
    """Test DiarizedSegment dataclass."""
    segment = DiarizedSegment(start=0.0, end=5.0, text="Test", speaker="SPEAKER_01")

    assert segment.start == 0.0
    assert segment.end == 5.0
    assert segment.text == "Test"
    assert segment.speaker == "SPEAKER_01"
    assert segment.words == []


def test_word_segment_dataclass():
    """Test WordSegment dataclass."""
    word = WordSegment(start=0.0, end=1.0, word="Hello", speaker="SPEAKER_01")

    assert word.start == 0.0
    assert word.end == 1.0
    assert word.word == "Hello"
    assert word.speaker == "SPEAKER_01"


class TestRenumberSpeaker:
    """Tests for _renumber_speaker edge cases."""

    def test_standard_renumber(self):
        """Test standard SPEAKER_XX -> SPEAKER_XX+1."""
        assert _renumber_speaker("SPEAKER_00") == "SPEAKER_01"
        assert _renumber_speaker("SPEAKER_01") == "SPEAKER_02"
        assert _renumber_speaker("SPEAKER_09") == "SPEAKER_10"

    def test_non_matching_pattern(self):
        """Test speaker names that don't match SPEAKER_XX pattern."""
        assert _renumber_speaker("Alice") == "Alice"
        assert _renumber_speaker("UNKNOWN") == "UNKNOWN"
        assert _renumber_speaker("speaker_00") == "speaker_00"  # lowercase

    def test_empty_string(self):
        """Test empty speaker string."""
        assert _renumber_speaker("") == ""

    def test_already_renumbered(self):
        """Test that already-renumbered speakers get renumbered again."""
        # This is expected behavior — _renumber_speaker always adds 1
        assert _renumber_speaker("SPEAKER_01") == "SPEAKER_02"

    def test_malformed_speaker_prefix(self):
        """Test SPEAKER_ prefix with non-numeric suffix."""
        assert _renumber_speaker("SPEAKER_") == "SPEAKER_"
        assert _renumber_speaker("SPEAKER_abc") == "SPEAKER_abc"


class TestDiarizerCloseAndContextManager:
    """Tests for close() and context manager protocol."""

    def test_close_releases_model(self):
        """Test that close() sets _model to None."""
        diarizer = Diarizer(model_name="base", hf_token="test_token")
        diarizer._model = MagicMock()  # Simulate loaded model
        diarizer._actual_device = "cpu"

        diarizer.close()

        assert diarizer._model is None
        assert diarizer._actual_device is None  # type: ignore[comparison-overlap]

    def test_context_manager_protocol(self):
        """Test __enter__/__exit__ context manager."""
        diarizer = Diarizer(model_name="base", hf_token="test_token")
        diarizer._model = MagicMock()

        with diarizer as d:
            assert d is diarizer
            assert d._model is not None

        # After exiting context, model should be released
        assert diarizer._model is None
