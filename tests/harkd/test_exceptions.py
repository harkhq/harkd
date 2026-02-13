"""Tests for harkd exceptions."""

import pytest

from harkd.exceptions import (
    HarkdError,
    InvalidStateError,
    NoActiveRecordingError,
    RecordingInProgressError,
    RecordingNotFoundError,
    StorageError,
    VoiceProfileNotFoundError,
)


def test_harkd_error_base():
    """Test base HarkdError."""
    err = HarkdError(
        message="Test error",
        code="TEST_ERROR",
        details={"key": "value"},
    )
    assert err.message == "Test error"
    assert err.code == "TEST_ERROR"
    assert err.details == {"key": "value"}
    assert str(err) == "Test error"


def test_harkd_error_no_details():
    """Test HarkdError without details."""
    err = HarkdError(message="Test", code="TEST")
    assert err.details == {}


def test_recording_in_progress_error():
    """Test RecordingInProgressError."""
    err = RecordingInProgressError(active_recording_id="rec-123")
    assert err.code == "RECORDING_IN_PROGRESS"
    assert err.message == "A recording is already in progress"
    assert err.details["active_recording_id"] == "rec-123"
    assert isinstance(err, HarkdError)


def test_no_active_recording_error():
    """Test NoActiveRecordingError."""
    err = NoActiveRecordingError()
    assert err.code == "NO_ACTIVE_RECORDING"
    assert err.message == "No active recording"
    assert err.details == {}


def test_recording_not_found_error():
    """Test RecordingNotFoundError."""
    err = RecordingNotFoundError(recording_id="rec-456")
    assert err.code == "RECORDING_NOT_FOUND"
    assert "rec-456" in err.message
    assert err.details["recording_id"] == "rec-456"


def test_voice_profile_not_found_error():
    """Test VoiceProfileNotFoundError."""
    err = VoiceProfileNotFoundError(profile_id="prof-789")
    assert err.code == "VOICE_PROFILE_NOT_FOUND"
    assert "prof-789" in err.message
    assert err.details["profile_id"] == "prof-789"


def test_storage_error():
    """Test StorageError."""
    err = StorageError(
        message="Failed to write file",
        details={"path": "/tmp/test.json"},
    )
    assert err.code == "STORAGE_ERROR"
    assert err.message == "Failed to write file"
    assert err.details["path"] == "/tmp/test.json"


def test_storage_error_no_details():
    """Test StorageError without details."""
    err = StorageError(message="Generic error")
    assert err.details == {}


def test_invalid_state_error():
    """Test InvalidStateError."""
    err = InvalidStateError(
        message="Cannot stop recording in processing state",
        current_state="processing",
        expected_state="recording",
    )
    assert err.code == "INVALID_STATE"
    assert "Cannot stop" in err.message
    assert err.details["current_state"] == "processing"
    assert err.details["expected_state"] == "recording"


def test_exceptions_are_catchable():
    """Test that specific exceptions can be caught as HarkdError."""
    try:
        raise RecordingNotFoundError("test-id")
    except HarkdError as e:
        assert e.code == "RECORDING_NOT_FOUND"
    else:
        pytest.fail("Exception should have been caught")


def test_exceptions_preserve_inheritance():
    """Test that exceptions maintain proper inheritance chain."""
    err = RecordingInProgressError("test-id")
    assert isinstance(err, HarkdError)
    assert isinstance(err, Exception)
    assert isinstance(err, BaseException)
