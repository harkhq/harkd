"""Tests for recording API models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from harkd.api.models.recording import (
    ProcessingStage,
    RecordingCreate,
    RecordingListItem,
    RecordingListResponse,
    RecordingOverrides,
    RecordingResponse,
    RecordingSettings,
    RecordingStatus,
    RecordingUpdate,
    SegmentModel,
    WordModel,
)


class TestRecordingStatus:
    """Test RecordingStatus enum."""

    def test_status_values(self):
        """Test enum has correct values."""
        assert RecordingStatus.RECORDING == "recording"
        assert RecordingStatus.PROCESSING == "processing"
        assert RecordingStatus.COMPLETE == "complete"
        assert RecordingStatus.ERROR == "error"


class TestProcessingStage:
    """Test ProcessingStage enum."""

    def test_stage_values(self):
        """Test enum has correct values."""
        assert ProcessingStage.PREPROCESSING == "preprocessing"
        assert ProcessingStage.TRANSCRIPTION == "transcription"
        assert ProcessingStage.DIARIZATION == "diarization"


class TestWordModel:
    """Test WordModel."""

    def test_word_model_minimal(self):
        """Test creating word with minimal fields."""
        word = WordModel(start=0.0, end=1.0, word="hello")
        assert word.start == 0.0
        assert word.end == 1.0
        assert word.word == "hello"
        assert word.speaker is None

    def test_word_model_with_speaker(self):
        """Test creating word with speaker."""
        word = WordModel(start=0.0, end=1.0, word="hello", speaker="Alice")
        assert word.speaker == "Alice"

    def test_word_model_end_before_start(self):
        """Test validation fails when end < start."""
        with pytest.raises(ValidationError) as exc_info:
            WordModel(start=2.0, end=1.0, word="hello")
        assert "end must be >= start" in str(exc_info.value)

    def test_word_model_negative_start(self):
        """Test validation fails for negative start time."""
        with pytest.raises(ValidationError):
            WordModel(start=-1.0, end=1.0, word="hello")


class TestSegmentModel:
    """Test SegmentModel."""

    def test_segment_model_minimal(self):
        """Test creating segment with minimal fields."""
        segment = SegmentModel(start=0.0, end=5.0, text="Hello world")
        assert segment.start == 0.0
        assert segment.end == 5.0
        assert segment.text == "Hello world"
        assert segment.speaker is None
        assert segment.words == []

    def test_segment_model_with_words(self):
        """Test segment with word-level timestamps."""
        words = [
            WordModel(start=0.0, end=1.0, word="Hello"),
            WordModel(start=1.0, end=2.0, word="world"),
        ]
        segment = SegmentModel(start=0.0, end=2.0, text="Hello world", words=words)
        assert len(segment.words) == 2
        assert segment.words[0].word == "Hello"

    def test_segment_model_end_before_start(self):
        """Test validation fails when end < start."""
        with pytest.raises(ValidationError) as exc_info:
            SegmentModel(start=5.0, end=1.0, text="test")
        assert "end must be >= start" in str(exc_info.value)


class TestRecordingSettings:
    """Test RecordingSettings."""

    def test_settings_defaults(self):
        """Test default settings values."""
        settings = RecordingSettings()
        assert settings.input_source == "mic"
        assert settings.model == "base"
        assert settings.language == "auto"
        assert settings.diarization is True
        assert settings.noise_reduction is True
        assert settings.normalization is True
        assert settings.word_timestamps is False

    def test_settings_custom(self):
        """Test creating settings with custom values."""
        settings = RecordingSettings(
            input_source="speaker",
            model="large-v3",
            language="en",
            diarization=False,
            noise_reduction=False,
            normalization=False,
            word_timestamps=True,
        )
        assert settings.input_source == "speaker"
        assert settings.model == "large-v3"
        assert settings.language == "en"
        assert settings.diarization is False
        assert settings.word_timestamps is True

    def test_settings_invalid_model(self):
        """Test validation fails for invalid model."""
        with pytest.raises(ValidationError) as exc_info:
            RecordingSettings(model="invalid-model")
        assert "Invalid model" in str(exc_info.value)

    def test_settings_valid_models(self):
        """Test all valid model names are accepted."""
        valid_models = [
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
        ]
        for model in valid_models:
            settings = RecordingSettings(model=model)
            assert settings.model == model

    def test_settings_invalid_input_source(self):
        """Test validation fails for invalid input source."""
        with pytest.raises(ValidationError):
            RecordingSettings(input_source="invalid")


class TestRecordingOverrides:
    """Test RecordingOverrides."""

    def test_overrides_all_none(self):
        """Test creating overrides with no values set."""
        overrides = RecordingOverrides()
        assert overrides.language is None
        assert overrides.input_source is None
        assert overrides.diarization is None
        assert overrides.noise_reduction is None
        assert overrides.normalization is None
        assert overrides.word_timestamps is None

    def test_overrides_partial(self):
        """Test creating overrides with some values set."""
        overrides = RecordingOverrides(language="en", diarization=False)
        assert overrides.language == "en"
        assert overrides.diarization is False
        assert overrides.input_source is None  # Not overridden

    def test_overrides_exclude_none(self):
        """Test model_dump(exclude_none=True) returns only set fields."""
        overrides = RecordingOverrides(language="en", diarization=False)
        dumped = overrides.model_dump(exclude_none=True)
        assert dumped == {"language": "en", "diarization": False}

    def test_overrides_no_model_field(self):
        """Test that model cannot be set on overrides."""
        # model is not a field on RecordingOverrides
        assert (
            not hasattr(RecordingOverrides.model_fields, "model")
            or "model" not in RecordingOverrides.model_fields
        )

    def test_overrides_invalid_input_source(self):
        """Test validation fails for invalid input source."""
        with pytest.raises(ValidationError):
            RecordingOverrides(input_source="invalid")


class TestRecordingCreate:
    """Test RecordingCreate."""

    def test_create_minimal(self):
        """Test creating request with minimal fields."""
        create = RecordingCreate()
        assert create.title is None
        assert create.settings is None

    def test_create_with_title(self):
        """Test creating request with title."""
        create = RecordingCreate(title="Team standup")
        assert create.title == "Team standup"

    def test_create_with_overrides(self):
        """Test creating request with setting overrides."""
        overrides = RecordingOverrides(language="en", diarization=False)
        create = RecordingCreate(title="Meeting", settings=overrides)
        assert create.title == "Meeting"
        assert create.settings is not None
        assert create.settings.language == "en"
        assert create.settings.diarization is False

    def test_create_title_too_long(self):
        """Test validation fails for title > 200 chars."""
        long_title = "x" * 201
        with pytest.raises(ValidationError):
            RecordingCreate(title=long_title)


class TestRecordingUpdate:
    """Test RecordingUpdate."""

    def test_update_empty(self):
        """Test creating empty update."""
        update = RecordingUpdate()
        assert update.title is None
        assert update.speakers is None

    def test_update_title(self):
        """Test updating title."""
        update = RecordingUpdate(title="Updated title")
        assert update.title == "Updated title"

    def test_update_speakers(self):
        """Test updating speaker mapping."""
        update = RecordingUpdate(speakers={"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})
        assert update.speakers is not None
        assert update.speakers["SPEAKER_00"] == "Alice"
        assert update.speakers["SPEAKER_01"] == "Bob"

    def test_update_both(self):
        """Test updating title and speakers."""
        update = RecordingUpdate(title="Updated", speakers={"SPEAKER_00": "Alice"})
        assert update.title == "Updated"
        assert update.speakers is not None
        assert update.speakers["SPEAKER_00"] == "Alice"


class TestRecordingResponse:
    """Test RecordingResponse."""

    def test_response_recording_status(self):
        """Test response for recording in progress."""
        response = RecordingResponse(
            id="rec-123",
            status=RecordingStatus.RECORDING,
            created_at=datetime(2026, 1, 15, 10, 30, 0),
            title="Test recording",
            duration=45.3,
            audio_level=0.42,
            settings=RecordingSettings(),
        )
        assert response.status == RecordingStatus.RECORDING
        assert response.duration == 45.3
        assert response.audio_level == 0.42
        assert response.processing_stage is None

    def test_response_processing_status(self):
        """Test response for processing."""
        response = RecordingResponse(
            id="rec-123",
            status=RecordingStatus.PROCESSING,
            created_at=datetime(2026, 1, 15, 10, 30, 0),
            title="Test recording",
            duration=120.5,
            processing_stage=ProcessingStage.TRANSCRIPTION,
            processing_progress=0.65,
            settings=RecordingSettings(),
        )
        assert response.status == RecordingStatus.PROCESSING
        assert response.processing_stage == ProcessingStage.TRANSCRIPTION
        assert response.processing_progress == 0.65

    def test_response_complete_status(self):
        """Test response for completed recording."""
        segments = [
            SegmentModel(start=0.0, end=5.0, text="Hello", speaker="Alice"),
        ]
        response = RecordingResponse(
            id="rec-123",
            status=RecordingStatus.COMPLETE,
            created_at=datetime(2026, 1, 15, 10, 30, 0),
            title="Test recording",
            duration=120.5,
            input_source="mic",
            model="base",
            language="en",
            language_confidence=0.98,
            diarized=True,
            speakers=["Alice", "Bob"],
            segments=segments,
            transcript="Hello",
            settings=RecordingSettings(),
        )
        assert response.status == RecordingStatus.COMPLETE
        assert response.language == "en"
        assert response.language_confidence == 0.98
        assert response.speakers is not None
        assert response.segments is not None
        assert len(response.speakers) == 2
        assert len(response.segments) == 1

    def test_response_invalid_audio_level(self):
        """Test validation fails for audio level > 1."""
        with pytest.raises(ValidationError):
            RecordingResponse(
                id="rec-123",
                status=RecordingStatus.RECORDING,
                created_at=datetime.now(),
                title="Test",
                duration=10.0,
                audio_level=1.5,
                settings=RecordingSettings(),
            )

    def test_response_future_ai_fields(self):
        """Test future AI feature fields are present."""
        response = RecordingResponse(
            id="rec-123",
            status=RecordingStatus.COMPLETE,
            created_at=datetime.now(),
            title="Test",
            duration=10.0,
            settings=RecordingSettings(),
        )
        assert response.tags == []
        assert response.executive_summary == []
        assert response.meeting_notes == []
        assert response.tasks == []
        assert response.decisions == []


class TestRecordingListItem:
    """Test RecordingListItem."""

    def test_list_item(self):
        """Test creating list item."""
        item = RecordingListItem(
            id="rec-123",
            title="Team standup",
            created_at=datetime(2026, 1, 15, 10, 30, 0),
            duration=120.5,
            status=RecordingStatus.COMPLETE,
            speakers=["Alice", "Bob"],
            language="en",
        )
        assert item.id == "rec-123"
        assert item.title == "Team standup"
        assert item.duration == 120.5
        assert item.status == RecordingStatus.COMPLETE
        assert len(item.speakers) == 2


class TestRecordingListResponse:
    """Test RecordingListResponse."""

    def test_list_response(self):
        """Test creating list response."""
        items = [
            RecordingListItem(
                id="rec-1",
                title="Recording 1",
                created_at=datetime.now(),
                duration=60.0,
                status=RecordingStatus.COMPLETE,
            ),
            RecordingListItem(
                id="rec-2",
                title="Recording 2",
                created_at=datetime.now(),
                duration=90.0,
                status=RecordingStatus.COMPLETE,
            ),
        ]
        response = RecordingListResponse(total=2, limit=50, offset=0, recordings=items)
        assert response.total == 2
        assert response.limit == 50
        assert response.offset == 0
        assert len(response.recordings) == 2

    def test_list_response_empty(self):
        """Test empty list response."""
        response = RecordingListResponse(total=0, limit=50, offset=0, recordings=[])
        assert response.total == 0
        assert len(response.recordings) == 0

    def test_list_response_pagination(self):
        """Test list response with pagination."""
        response = RecordingListResponse(total=100, limit=10, offset=20, recordings=[])
        assert response.total == 100
        assert response.limit == 10
        assert response.offset == 20

    def test_list_response_invalid_limit(self):
        """Test validation fails for limit > 100."""
        with pytest.raises(ValidationError):
            RecordingListResponse(total=0, limit=101, offset=0, recordings=[])

    def test_list_response_invalid_offset(self):
        """Test validation fails for negative offset."""
        with pytest.raises(ValidationError):
            RecordingListResponse(total=0, limit=50, offset=-1, recordings=[])
