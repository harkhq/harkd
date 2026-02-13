"""Custom exceptions for harkd."""

from typing import Any

__all__ = [
    "HarkdError",
    "RecordingInProgressError",
    "NoActiveRecordingError",
    "RecordingNotFoundError",
    "VoiceProfileNotFoundError",
    "StorageError",
    "InvalidStateError",
    "NoMicrophoneError",
    "NoLoopbackDeviceError",
]


class HarkdError(Exception):
    """Base exception for harkd.

    All harkd exceptions include:
    - message: Human-readable error message
    - code: Machine-readable error code
    - details: Optional dict with additional context
    """

    def __init__(self, message: str, code: str, details: dict[str, Any] | None = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)


class RecordingInProgressError(HarkdError):
    """Raised when attempting to start a recording while one is already active."""

    def __init__(self, active_recording_id: str):
        super().__init__(
            message="A recording is already in progress",
            code="RECORDING_IN_PROGRESS",
            details={"active_recording_id": active_recording_id},
        )


class NoActiveRecordingError(HarkdError):
    """Raised when attempting to stop/cancel but no recording is active."""

    def __init__(self):
        super().__init__(
            message="No active recording",
            code="NO_ACTIVE_RECORDING",
        )


class RecordingNotFoundError(HarkdError):
    """Raised when a recording ID doesn't exist."""

    def __init__(self, recording_id: str):
        super().__init__(
            message=f"Recording not found: {recording_id}",
            code="RECORDING_NOT_FOUND",
            details={"recording_id": recording_id},
        )


class VoiceProfileNotFoundError(HarkdError):
    """Raised when a voice profile ID doesn't exist."""

    def __init__(self, profile_id: str):
        super().__init__(
            message=f"Voice profile not found: {profile_id}",
            code="VOICE_PROFILE_NOT_FOUND",
            details={"profile_id": profile_id},
        )


class StorageError(HarkdError):
    """Raised when filesystem/storage operations fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            code="STORAGE_ERROR",
            details=details,
        )


class InvalidStateError(HarkdError):
    """Raised when an operation is not valid in the current state."""

    def __init__(self, message: str, current_state: str, expected_state: str):
        super().__init__(
            message=message,
            code="INVALID_STATE",
            details={
                "current_state": current_state,
                "expected_state": expected_state,
            },
        )


class NoMicrophoneError(HarkdError):
    """Raised when microphone input is requested but no microphone device is found."""

    def __init__(self, message: str = "No microphone device found"):
        super().__init__(
            message=message,
            code="NO_MICROPHONE",
        )


class NoLoopbackDeviceError(HarkdError):
    """Raised when loopback/speaker input is requested but no loopback device is found."""

    def __init__(self, message: str = "No loopback device found"):
        super().__init__(
            message=message,
            code="NO_LOOPBACK_DEVICE",
        )
