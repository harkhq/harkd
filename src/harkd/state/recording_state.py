"""Recording state management for tracking active recordings."""

import asyncio
import functools
from datetime import UTC, datetime

from harkd.exceptions import NoActiveRecordingError, RecordingInProgressError

__all__ = ["RecordingState", "get_recording_state"]


class RecordingState:
    """Singleton state manager for active recordings.

    Tracks the currently active recording to ensure only one
    recording can be active at a time.
    """

    def __init__(self):
        """Initialize recording state."""
        self._active_recording_id: str | None = None
        self._start_time: datetime | None = None
        self._lock = asyncio.Lock()

    @property
    def is_recording(self) -> bool:
        """Check if a recording is currently active."""
        return self._active_recording_id is not None

    @property
    def active_recording_id(self) -> str | None:
        """Get the ID of the active recording."""
        return self._active_recording_id

    @property
    def start_time(self) -> datetime | None:
        """Get the start time of the active recording."""
        return self._start_time

    @property
    def duration(self) -> float:
        """Get the current duration of the active recording in seconds.

        Note: This property is not locked for performance. For atomic access,
        use get_status() instead. The result may be slightly stale but safe.
        """
        start_time = self._start_time  # Atomic read in CPython
        if start_time is None:
            return 0.0
        return (datetime.now(UTC) - start_time).total_seconds()

    async def get_status(self) -> dict:
        """Get current recording status atomically.

        Returns:
            Dict with is_recording, recording_id (if active), and duration
        """
        async with self._lock:
            if self._active_recording_id is None:
                return {"is_recording": False, "duration": 0.0}
            start_time = self._start_time
            if start_time is None:
                raise RuntimeError("start_time is None while recording is active")
            return {
                "is_recording": True,
                "recording_id": self._active_recording_id,
                "duration": (datetime.now(UTC) - start_time).total_seconds(),
            }

    async def start(self, recording_id: str) -> None:
        """Start tracking a new recording.

        Args:
            recording_id: ID of the recording to track

        Raises:
            RecordingInProgressError: If a recording is already active
        """
        async with self._lock:
            if self._active_recording_id is not None:
                raise RecordingInProgressError(self._active_recording_id)
            self._active_recording_id = recording_id
            self._start_time = datetime.now(UTC)

    async def stop(self) -> tuple[str, datetime]:
        """Stop tracking the active recording.

        Returns:
            Tuple of (recording_id, start_time)

        Raises:
            NoActiveRecordingError: If no recording is active
        """
        async with self._lock:
            if self._active_recording_id is None:
                raise NoActiveRecordingError()

            recording_id = self._active_recording_id
            start_time = self._start_time
            if start_time is None:
                raise RuntimeError("start_time is None while recording is active")

            self._active_recording_id = None
            self._start_time = None

            return recording_id, start_time

    async def cancel(self) -> None:
        """Cancel the active recording without returning info.

        Raises:
            NoActiveRecordingError: If no recording is active
        """
        async with self._lock:
            if self._active_recording_id is None:
                raise NoActiveRecordingError()

            self._active_recording_id = None
            self._start_time = None


@functools.lru_cache(maxsize=1)
def get_recording_state() -> RecordingState:
    """Get or create the global recording state singleton.

    Returns:
        Recording state instance
    """
    return RecordingState()
