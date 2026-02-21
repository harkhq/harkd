"""Custom exceptions for harkd."""

from typing import Any

__all__ = [
    "HarkdError",
    "RecordingInProgressError",
    "NoActiveRecordingError",
    "RecordingNotFoundError",
    "VoiceProfileNotFoundError",
    "ChatThreadNotFoundError",
    "LLMNotConfiguredError",
    "StorageError",
    "InvalidStateError",
    "NoMicrophoneError",
    "NoLoopbackDeviceError",
    "RetryNotAllowedError",
    "RemoteTranscriptionError",
    "InfraProvisioningError",
    "InfraTeardownError",
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


class ChatThreadNotFoundError(HarkdError):
    """Raised when a chat thread ID doesn't exist."""

    def __init__(self, thread_id: str):
        super().__init__(
            message=f"Chat thread not found: {thread_id}",
            code="CHAT_THREAD_NOT_FOUND",
            details={"thread_id": thread_id},
        )


class LLMNotConfiguredError(HarkdError):
    """Raised when LLM features are used but not configured."""

    def __init__(self):
        super().__init__(
            message="LLM features are not enabled. Set llm.enabled=true in configuration.",
            code="LLM_NOT_CONFIGURED",
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


class RetryNotAllowedError(HarkdError):
    """Raised when retry is not allowed for a recording."""

    def __init__(self, recording_id: str, reason: str):
        super().__init__(
            message=f"Cannot retry recording {recording_id}: {reason}",
            code="RETRY_NOT_ALLOWED",
            details={"recording_id": recording_id, "reason": reason},
        )


class RemoteTranscriptionError(HarkdError):
    """Raised when remote transcription backend fails."""

    def __init__(self, message: str, provider: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            code="REMOTE_TRANSCRIPTION_ERROR",
            details={"provider": provider, **(details or {})},
        )


class InfraProvisioningError(HarkdError):
    """Raised when infrastructure provisioning fails."""

    def __init__(self, message: str, provider: str = "", details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            code="INFRA_PROVISIONING_ERROR",
            details={"provider": provider, **(details or {})},
        )


class InfraTeardownError(HarkdError):
    """Raised when infrastructure teardown fails."""

    def __init__(self, message: str, provider: str = "", details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            code="INFRA_TEARDOWN_ERROR",
            details={"provider": provider, **(details or {})},
        )
