"""API request and response models."""

from harkd.api.models.error import ErrorDetail, ErrorResponse
from harkd.api.models.health import HealthResponse
from harkd.api.models.recording import (
    ActiveRecordingUpdate,
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
from harkd.api.models.settings import Settings
from harkd.api.models.voice_profile import (
    VoiceEmbedding,
    VoiceProfile,
    VoiceProfileCreate,
    VoiceProfileDetail,
    VoiceProfileListResponse,
)

__all__ = [
    # Error models
    "ErrorDetail",
    "ErrorResponse",
    # Health models
    "HealthResponse",
    # Recording models
    "ActiveRecordingUpdate",
    "ProcessingStage",
    "RecordingCreate",
    "RecordingListItem",
    "RecordingListResponse",
    "RecordingResponse",
    "RecordingSettings",
    "RecordingStatus",
    "RecordingUpdate",
    "SegmentModel",
    "WordModel",
    "RecordingOverrides",
    # Settings models
    "Settings",
    # Voice profile models
    "VoiceEmbedding",
    "VoiceProfile",
    "VoiceProfileCreate",
    "VoiceProfileDetail",
    "VoiceProfileListResponse",
]
