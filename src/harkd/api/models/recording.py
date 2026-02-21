"""API models for recordings."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "RecordingStatus",
    "ProcessingStage",
    "WordModel",
    "SegmentModel",
    "RecordingSettings",
    "RecordingOverrides",
    "RetryOverrides",
    "RecordingCreate",
    "RecordingUpdate",
    "ActiveRecordingUpdate",
    "RecordingResponse",
    "RecordingListItem",
    "RecordingListResponse",
]


class RecordingStatus(str, Enum):
    """Recording status enum."""

    RECORDING = "recording"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


class ProcessingStage(str, Enum):
    """Processing stage enum."""

    PREPROCESSING = "preprocessing"
    TRANSCRIPTION = "transcription"
    DIARIZATION = "diarization"
    MEETING_MINUTES = "meeting_minutes"


class WordModel(BaseModel):
    """Word segment with timing."""

    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    word: str = Field(..., description="Word text")
    speaker: str | None = Field(None, description="Speaker label")

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        """Validate end is after start."""
        if "start" in info.data and v < info.data["start"]:
            raise ValueError("end must be >= start")
        return v


class SegmentModel(BaseModel):
    """Transcript segment."""

    start: float = Field(..., ge=0)
    end: float = Field(..., ge=0)
    text: str
    speaker: str | None = None
    words: list[WordModel] = Field(default_factory=list)

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        """Validate end is after start."""
        if "start" in info.data and v < info.data["start"]:
            raise ValueError("end must be >= start")
        return v


class RecordingSettings(BaseModel):
    """Settings for a recording session."""

    mic_enabled: bool = Field(default=True, description="Microphone input enabled")
    speaker_enabled: bool = Field(default=True, description="Speaker input enabled")
    model: str = Field(default="base", description="Whisper model name")
    language: str = Field(default="auto", description="Language code or 'auto'")
    diarization: bool = Field(default=True, description="Enable speaker diarization")
    noise_reduction: bool = Field(default=True, description="Enable noise reduction")
    normalization: bool = Field(default=True, description="Enable audio normalization")
    mic_gain: float = Field(
        default=2.0, ge=0.1, le=10.0, description="Microphone gain multiplier (1.0 = no gain)"
    )
    word_timestamps: bool = Field(default=False, description="Include word-level timestamps")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate Whisper model name."""
        valid = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if v not in valid:
            raise ValueError(f"Invalid model. Must be one of: {valid}")
        return v


class RecordingOverrides(BaseModel):
    """Per-recording setting overrides.

    Only light settings can be overridden per-recording.
    model is daemon-level only (requires restart to change).
    All fields are optional — None means "use daemon default".
    """

    language: str | None = Field(None, description="Language code or 'auto'")
    mic_enabled: bool | None = Field(None, description="Microphone input enabled")
    speaker_enabled: bool | None = Field(None, description="Speaker input enabled")
    diarization: bool | None = Field(None, description="Enable speaker diarization")
    noise_reduction: bool | None = Field(None, description="Enable noise reduction")
    normalization: bool | None = Field(None, description="Enable audio normalization")
    mic_gain: float | None = Field(
        None, ge=0.1, le=10.0, description="Microphone gain multiplier (1.0 = no gain)"
    )
    word_timestamps: bool | None = Field(None, description="Include word-level timestamps")

    @model_validator(mode="after")
    def at_least_one_input(self) -> "RecordingOverrides":
        """Validate that at least one input is enabled when both are explicitly set."""
        if self.mic_enabled is False and self.speaker_enabled is False:
            raise ValueError("At least one input must be enabled")
        return self


class RetryOverrides(BaseModel):
    """Diarization overrides for retry/re-diarization."""

    num_speakers: int | None = Field(
        None, ge=1, description="Exact speaker count hint for diarization"
    )
    min_speakers: int | None = Field(None, ge=1, description="Minimum speakers")
    max_speakers: int | None = Field(None, ge=1, description="Maximum speakers")
    clustering_threshold: float | None = Field(
        None, ge=0.0, le=2.0, description="Lower = merge more aggressively, fewer speakers"
    )


class RecordingCreate(BaseModel):
    """Request to create/start a new recording."""

    title: str | None = Field(None, max_length=200, description="Optional recording title")
    settings: RecordingOverrides | None = Field(
        None, description="Per-recording setting overrides (None = use daemon defaults)"
    )


class RecordingUpdate(BaseModel):
    """Request to update a recording."""

    title: str | None = Field(None, max_length=200)
    tags: list[str] | None = None
    tasks: list[dict[str, Any]] | None = None
    decisions: list[str] | None = None
    speakers: dict[str, str] | None = Field(
        None, description="Speaker label mapping: {SPEAKER_00: 'Alice'}"
    )
    speaker_profile_ids: dict[str, str] | None = Field(
        None,
        description="Map speaker labels to existing voice profile IDs",
    )
    create_voice_profiles: bool = Field(
        False, description="Create voice profiles for named speakers"
    )


class ActiveRecordingUpdate(BaseModel):
    """Request to update the active recording (title, input toggles)."""

    title: str | None = Field(None, max_length=200)
    mic_enabled: bool | None = None
    speaker_enabled: bool | None = None


class RecordingResponse(BaseModel):
    """Full recording response."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="Unique recording ID")
    status: RecordingStatus
    created_at: datetime
    title: str
    duration: float = Field(..., ge=0)

    # While recording:
    mic_enabled: bool | None = None
    speaker_enabled: bool | None = None
    mic_level: float | None = Field(None, ge=0, le=1, description="Current mic audio level (0-1)")
    speaker_level: float | None = Field(
        None, ge=0, le=1, description="Current speaker audio level (0-1)"
    )

    # While processing:
    processing_stage: ProcessingStage | None = None
    processing_progress: float | None = Field(None, ge=0, le=1)
    processing_started_at: datetime | None = Field(
        None, description="When processing began (ISO 8601)"
    )
    retry_count: int = Field(default=0, description="Number of processing attempts")
    last_error: str | None = Field(default=None, description="Last error message")

    # Processing metrics:
    processing_duration: float | None = Field(
        None, ge=0, description="Wall-clock processing time in seconds"
    )

    # When complete:
    model: str | None = None
    language: str | None = None
    language_confidence: float | None = Field(None, ge=0, le=1)
    diarized: bool | None = None
    speakers: list[str] | None = None
    speaker_embeddings: dict[str, list[float]] | None = None
    speaker_profiles: dict[str, str] | None = None
    segments: list[SegmentModel] | None = None
    transcript: str | None = None

    # Future AI features (structure ready):
    tags: list[str] = Field(default_factory=list)
    executive_summary: list[str] = Field(default_factory=list)
    meeting_notes: list[dict[str, Any]] = Field(default_factory=list)
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)

    settings: RecordingSettings | dict[str, Any]


class RecordingListItem(BaseModel):
    """Minimal recording info for list views."""

    id: str
    title: str
    created_at: datetime
    duration: float = Field(..., ge=0)
    status: RecordingStatus
    speakers: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    language: str | None = None


class RecordingListResponse(BaseModel):
    """Paginated list of recordings."""

    total: int = Field(..., ge=0)
    limit: int = Field(..., ge=1, le=100)
    offset: int = Field(..., ge=0)
    recordings: list[RecordingListItem]
