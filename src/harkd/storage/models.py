"""Internal storage models for harkd.

These models are separate from API models to allow independent evolution.
They handle JSON serialization/deserialization for filesystem storage.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "SpeakerEmbeddings",
    "StorageChatMessage",
    "StorageChatScope",
    "StorageChatThread",
    "StorageRecording",
    "StorageVoiceProfile",
]

SpeakerEmbeddings = dict[str, list[float]]


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
    processing_started_at: datetime | None = None

    # Retry state (persisted across restarts)
    retry_count: int = Field(default=0, ge=0, description="Number of processing attempts")
    max_retries: int = Field(default=3, ge=0, description="Max retry attempts")
    last_error: str | None = Field(default=None, description="Last error message")
    last_error_at: datetime | None = Field(default=None, description="When the last error occurred")
    error_history: list[dict[str, Any]] = Field(
        default_factory=list, description="All error records"
    )

    # Processing metrics
    processing_duration: float | None = Field(
        None, ge=0, description="Wall-clock processing time in seconds"
    )

    # Complete recording data
    model: str | None = None
    language: str | None = None
    language_confidence: float | None = Field(None, ge=0, le=1)
    diarized: bool | None = None
    speakers: list[str] = Field(default_factory=list)
    speaker_embeddings: dict[str, list[float]] | None = Field(default=None)
    speaker_profiles: dict[str, str] | None = Field(
        default=None, description="Maps speaker label to voice_profile_id"
    )
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


class StorageChatMessage(BaseModel):
    """A single message in a chat thread."""

    id: str
    role: Literal["user", "assistant", "tool"]
    content: str
    timestamp: datetime
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class StorageChatScope(BaseModel):
    """Scope filters for a chat thread."""

    recording_ids: list[str] = Field(default_factory=list)
    date_from: datetime | None = None
    date_to: datetime | None = None
    speakers: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class StorageChatThread(BaseModel):
    """A chat conversation thread.

    Stored as JSON in: ~/.local/share/hark/chat-threads/{id}.json
    """

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    scope: StorageChatScope = Field(default_factory=StorageChatScope)
    messages: list[StorageChatMessage] = Field(default_factory=list)
    context_summary: str | None = None
    recent_message_count: int = Field(default=20, ge=1)

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
