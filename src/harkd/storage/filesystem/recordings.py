"""Filesystem-based recording storage implementation."""

import asyncio
import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path

from harkd.exceptions import RecordingNotFoundError, StorageError
from harkd.storage.base import RecordingStorage
from harkd.storage.models import StorageRecording

__all__ = ["FilesystemRecordingStorage"]

logger = logging.getLogger(__name__)


class FilesystemRecordingStorage(RecordingStorage):
    """
    Filesystem-based recording storage.

    Storage structure:
        {base_path}/recordings/
            {recording-id}/
                metadata.json
    """

    def __init__(self, base_path: Path):
        """Initialize filesystem storage.

        Args:
            base_path: Base directory for storage (e.g., ~/.local/share/hark)
        """
        self.base_path = base_path
        self.recordings_dir = base_path / "recordings"

    async def create(self, recording: StorageRecording) -> StorageRecording:
        """Create a new recording."""
        recording_dir = self.recordings_dir / recording.id

        try:
            # Create recording directory
            await asyncio.to_thread(recording_dir.mkdir, parents=True, exist_ok=False)

            # Write metadata
            await self._write_metadata(recording_dir, recording)

            return recording

        except FileExistsError as e:
            raise StorageError(
                f"Recording {recording.id} already exists",
                details={"recording_id": recording.id},
            ) from e
        except Exception as e:
            # Clean up on failure
            await asyncio.to_thread(self._cleanup_directory, recording_dir)
            raise StorageError(
                f"Failed to create recording: {e}",
                details={"recording_id": recording.id},
            ) from e

    async def get(self, recording_id: str) -> StorageRecording | None:
        """Get a recording by ID."""
        recording_dir = self.recordings_dir / recording_id
        metadata_file = recording_dir / "metadata.json"

        if not await asyncio.to_thread(metadata_file.exists):
            return None

        try:
            return await self._read_metadata(metadata_file)
        except Exception as e:
            raise StorageError(
                f"Failed to read recording {recording_id}: {e}",
                details={"recording_id": recording_id},
            ) from e

    async def list(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> list[StorageRecording]:
        """List recordings with pagination and filtering."""
        try:
            # Get all recording directories
            recordings_dir = self.recordings_dir
            if not await asyncio.to_thread(recordings_dir.exists):
                return []

            # Read all recordings inside a single thread call to avoid
            # blocking the event loop with per-directory sync I/O
            def _read_all_recordings() -> list[tuple[Path, bool, bool]]:
                result = []
                for rec_dir in recordings_dir.iterdir():
                    is_dir = rec_dir.is_dir()
                    metadata_file = rec_dir / "metadata.json"
                    has_metadata = metadata_file.exists() if is_dir else False
                    result.append((rec_dir, is_dir, has_metadata))
                return result

            dir_entries = await asyncio.to_thread(_read_all_recordings)

            recordings = []
            for rec_dir, is_dir, has_metadata in dir_entries:
                if not is_dir or not has_metadata:
                    continue

                metadata_file = rec_dir / "metadata.json"
                try:
                    recording = await self._read_metadata(metadata_file)
                    recordings.append(recording)
                except Exception:
                    logger.warning(f"Skipping corrupted recording metadata: {metadata_file}")
                    continue

            # Filter by status if provided
            if status:
                recordings = [r for r in recordings if r.status == status]

            # Sort by created_at descending (newest first)
            recordings.sort(key=lambda r: r.created_at, reverse=True)

            # Apply pagination (limit=0 means return all)
            if limit > 0:
                return recordings[offset : offset + limit]
            return recordings[offset:] if offset > 0 else recordings

        except Exception as e:
            raise StorageError(f"Failed to list recordings: {e}") from e

    async def update(self, recording: StorageRecording) -> StorageRecording:
        """Update an existing recording."""
        recording_dir = self.recordings_dir / recording.id

        if not await asyncio.to_thread(recording_dir.exists):
            raise RecordingNotFoundError(recording.id)

        try:
            await self._write_metadata(recording_dir, recording)
            return recording
        except Exception as e:
            raise StorageError(
                f"Failed to update recording {recording.id}: {e}",
                details={"recording_id": recording.id},
            ) from e

    async def delete(self, recording_id: str) -> None:
        """Delete a recording."""
        recording_dir = self.recordings_dir / recording_id

        if not await asyncio.to_thread(recording_dir.exists):
            raise RecordingNotFoundError(recording_id)

        try:
            await asyncio.to_thread(self._delete_directory, recording_dir)
        except Exception as e:
            raise StorageError(
                f"Failed to delete recording {recording_id}: {e}",
                details={"recording_id": recording_id},
            ) from e

    async def count(self, status: str | None = None) -> int:
        """Count recordings."""
        try:
            recordings_dir = self.recordings_dir
            if not await asyncio.to_thread(recordings_dir.exists):
                return 0

            if status is None:
                # Fast path: count directories directly
                def _count_dirs() -> int:
                    return sum(
                        1
                        for d in recordings_dir.iterdir()
                        if d.is_dir() and (d / "metadata.json").exists()
                    )

                return await asyncio.to_thread(_count_dirs)

            # Status filter requires reading metadata
            all_recordings = await self.list(limit=0, offset=0, status=status)
            return len(all_recordings)

        except Exception as e:
            raise StorageError(f"Failed to count recordings: {e}") from e

    # Helper methods

    async def _write_metadata(self, recording_dir: Path, recording: StorageRecording) -> None:
        """Write recording metadata to JSON file atomically.

        Uses write-to-temp-then-rename pattern for crash safety.
        """
        metadata_file = recording_dir / "metadata.json"
        data = recording.model_dump(mode="json")

        def write_atomic():
            # Write to temp file in same directory (same filesystem for atomic rename)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(recording_dir), suffix=".tmp", prefix=".metadata-"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                # Atomic rename on POSIX
                os.replace(tmp_path, str(metadata_file))
            except BaseException:
                # Clean up temp file on any failure
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise

        await asyncio.to_thread(write_atomic)

    async def _read_metadata(self, metadata_file: Path) -> StorageRecording:
        """Read recording metadata from JSON file."""

        def read():
            with open(metadata_file, encoding="utf-8") as f:
                return json.load(f)

        data = await asyncio.to_thread(read)
        return StorageRecording.model_validate(data)

    def _cleanup_directory(self, directory: Path) -> None:
        """Remove directory and contents (sync)."""
        import shutil

        if directory.exists():
            shutil.rmtree(directory)

    def _delete_directory(self, directory: Path) -> None:
        """Delete directory and contents (sync)."""
        import shutil

        shutil.rmtree(directory)
