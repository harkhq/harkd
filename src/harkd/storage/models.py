"""Internal storage models for harkd.

These models are separate from API models to allow independent evolution.
They handle JSON serialization/deserialization for filesystem storage.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "StorageRecording",
    "StorageVoiceProfile",
]


class StorageRecording(BaseModel):
    """
    Internal storage model for recordings.

    Stored as JSON in: ~/.local/share/hark/recordings/{id}/metadata.json
    """

    id: str
    status: Literal["recording", "processing", "complete", "error"]
    created_at: datetime
    title: str
    duration: float = Field(ge=0)

    # Recording state (while active/processing)
    mic_enabled: bool | None = None
    speaker_enabled: bool | None = None
    mic_level: float | None = Field(None, ge=0, le=1)
    speaker_level: float | None = Field(None, ge=0, le=1)
    processing_stage: (
        Literal["preprocessing", "transcription", "diarization", "meeting_minutes"] | None
    ) = None
    processing_progress: float | None = Field(None, ge=0, le=1)

    # Complete recording data
    model: str | None = None
    language: str | None = None
    language_confidence: float | None = Field(None, ge=0, le=1)
    diarized: bool | None = None
    speakers: list[str] = Field(default_factory=list)
    segments: list[dict[str, Any]] = Field(default_factory=list)
    transcript: str | None = None

    # Settings used for this recording
    settings: dict[str, Any]

    # Future features (empty for now)
    tags: list[str] = Field(default_factory=list)
    executive_summary: list[str] = Field(default_factory=list)
    meeting_notes: list[dict[str, Any]] = Field(default_factory=list)
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class StorageVoiceProfile(BaseModel):
    """
    Internal storage model for voice profiles.

    Stored as JSON in: ~/.local/share/hark/voice-profiles/{id}.json
    """

    id: str
    name: str
    created_at: datetime
    last_used: datetime | None = None
    clips: int = Field(default=0, ge=0)
    total_seconds: float = Field(default=0, ge=0)
    confidence: float = Field(default=0, ge=0, le=1)
    embeddings: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")
