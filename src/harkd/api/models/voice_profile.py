"""API models for voice profiles."""

from datetime import datetime

from pydantic import BaseModel, Field

__all__ = [
    "VoiceEmbedding",
    "VoiceProfile",
    "VoiceProfileDetail",
    "VoiceProfileCreate",
    "VoiceProfileListResponse",
    "SpeakerSegment",
    "UnassignedSpeaker",
    "UnassignedSpeakersResponse",
    "ProfileClip",
    "ProfileClipsResponse",
    "ClipAssignment",
]


class VoiceEmbedding(BaseModel):
    """Single voice embedding from a recording."""

    recording_id: str
    speaker_id: str
    timestamp: datetime
    vector: list[float] = Field(default_factory=list)


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


class SpeakerSegment(BaseModel):
    """A playable segment of speech."""

    start: float = Field(..., ge=0)
    end: float = Field(..., ge=0)
    text: str


class UnassignedSpeaker(BaseModel):
    """A speaker from a recording not yet assigned to any voice profile."""

    recording_id: str
    recording_title: str
    recording_created_at: datetime
    speaker_label: str
    segments: list[SpeakerSegment]


class UnassignedSpeakersResponse(BaseModel):
    """List of unassigned speakers across all recordings."""

    speakers: list[UnassignedSpeaker]


class ProfileClip(BaseModel):
    """A clip assigned to a voice profile."""

    recording_id: str
    recording_title: str
    recording_created_at: datetime
    speaker_label: str
    segments: list[SpeakerSegment]


class ProfileClipsResponse(BaseModel):
    """Clips associated with a voice profile."""

    profile_id: str
    profile_name: str
    clips: list[ProfileClip]


class ClipAssignment(BaseModel):
    """Request to assign a speaker clip to a profile."""

    recording_id: str
    speaker_label: str
