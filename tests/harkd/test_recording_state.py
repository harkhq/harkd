"""Tests for recording state management."""

import asyncio
from datetime import datetime

import pytest

from harkd.exceptions import NoActiveRecordingError, RecordingInProgressError
from harkd.state.recording_state import RecordingState, get_recording_state


@pytest.fixture
def state():
    """Create a fresh recording state for each test."""
    return RecordingState()


@pytest.mark.asyncio
async def test_initial_state(state):
    """Test initial state is not recording."""
    assert state.is_recording is False
    assert state.active_recording_id is None
    assert state.start_time is None
    assert state.duration == 0.0


@pytest.mark.asyncio
async def test_start_recording(state):
    """Test starting a recording."""
    await state.start("rec-123")

    assert state.is_recording is True
    assert state.active_recording_id == "rec-123"
    assert state.start_time is not None
    assert isinstance(state.start_time, datetime)


@pytest.mark.asyncio
async def test_start_recording_when_active(state):
    """Test that starting when already active raises error."""
    await state.start("rec-123")

    with pytest.raises(RecordingInProgressError) as exc_info:
        await state.start("rec-456")

    assert "already in progress" in str(exc_info.value)


@pytest.mark.asyncio
async def test_stop_recording(state):
    """Test stopping a recording."""
    await state.start("rec-123")
    recording_id, start_time = await state.stop()

    assert recording_id == "rec-123"
    assert isinstance(start_time, datetime)
    assert state.is_recording is False
    assert state.active_recording_id is None
    assert state.start_time is None


@pytest.mark.asyncio
async def test_stop_recording_when_not_active(state):
    """Test that stopping when not active raises error."""
    with pytest.raises(NoActiveRecordingError):
        await state.stop()


@pytest.mark.asyncio
async def test_cancel_recording(state):
    """Test canceling a recording."""
    await state.start("rec-123")
    await state.cancel()

    assert state.is_recording is False
    assert state.active_recording_id is None
    assert state.start_time is None


@pytest.mark.asyncio
async def test_cancel_recording_when_not_active(state):
    """Test that canceling when not active raises error."""
    with pytest.raises(NoActiveRecordingError):
        await state.cancel()


@pytest.mark.asyncio
async def test_duration_calculation(state):
    """Test that duration is calculated correctly."""
    await state.start("rec-123")

    # Wait a bit
    await asyncio.sleep(0.1)

    duration = state.duration
    assert duration > 0
    assert duration < 1  # Should be around 0.1 seconds


@pytest.mark.asyncio
async def test_duration_when_not_recording(state):
    """Test that duration is 0 when not recording."""
    assert state.duration == 0.0


@pytest.mark.asyncio
async def test_multiple_start_stop_cycles(state):
    """Test multiple start/stop cycles."""
    # First recording
    await state.start("rec-1")
    assert state.active_recording_id == "rec-1"
    await state.stop()
    assert state.is_recording is False  # type: ignore[comparison-overlap]

    # Second recording
    await state.start("rec-2")
    assert state.active_recording_id == "rec-2"
    await state.stop()
    assert state.is_recording is False  # type: ignore[comparison-overlap]

    # Third recording
    await state.start("rec-3")
    assert state.active_recording_id == "rec-3"
    assert state.is_recording is True  # type: ignore[comparison-overlap]


@pytest.mark.asyncio
async def test_get_recording_state_singleton():
    """Test that get_recording_state returns singleton."""
    state1 = get_recording_state()
    state2 = get_recording_state()

    assert state1 is state2


@pytest.mark.asyncio
async def test_concurrent_start_attempts(state):
    """Test that concurrent start attempts are properly serialized."""
    # Try to start two recordings concurrently
    results = await asyncio.gather(
        state.start("rec-1"),
        state.start("rec-2"),
        return_exceptions=True,
    )

    # One should succeed, one should fail
    errors = [r for r in results if isinstance(r, Exception)]
    successes = [r for r in results if r is None]

    assert len(errors) == 1
    assert len(successes) == 1
    assert isinstance(errors[0], RecordingInProgressError)
    assert "already in progress" in str(errors[0])
