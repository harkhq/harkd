"""API models for voice profiles."""

from datetime import datetime

from pydantic import BaseModel, Field

__all__ = [
    "VoiceEmbedding",
    "VoiceProfile",
    "VoiceProfileDetail",
    "VoiceProfileCreate",
    "VoiceProfileListResponse",
]


class VoiceEmbedding(BaseModel):
    """Single voice embedding from a recording."""

    recording_id: str
    speaker_id: str
    timestamp: datetime


class VoiceProfile(BaseModel):
    """Voice profile for speaker identification."""

    id: str
    name: str = Field(..., min_length=1, max_length=100)
    created_at: datetime
    last_used: datetime | None = None
    clips: int = Field(default=0, ge=0)
    total_seconds: float = Field(default=0, ge=0)
    confidence: float = Field(default=0, ge=0, le=1)


class VoiceProfileDetail(VoiceProfile):
    """Voice profile with embeddings."""

    embeddings: list[VoiceEmbedding] = Field(default_factory=list)


class VoiceProfileCreate(BaseModel):
    """Request to create new voice profile."""

    name: str = Field(..., min_length=1, max_length=100)


class VoiceProfileListResponse(BaseModel):
    """List of voice profiles."""

    profiles: list[VoiceProfile]
