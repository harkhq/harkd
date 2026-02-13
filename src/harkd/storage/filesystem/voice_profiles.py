"""Filesystem-based voice profile storage implementation."""

import asyncio
import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path

from harkd.exceptions import StorageError, VoiceProfileNotFoundError
from harkd.storage.base import VoiceProfileStorage
from harkd.storage.models import StorageVoiceProfile

__all__ = ["FilesystemVoiceProfileStorage"]

logger = logging.getLogger(__name__)


class FilesystemVoiceProfileStorage(VoiceProfileStorage):
    """
    Filesystem-based voice profile storage.

    Storage structure:
        {base_path}/voice-profiles/
            {profile-id}.json
    """

    def __init__(self, base_path: Path):
        """Initialize filesystem storage.

        Args:
            base_path: Base directory for storage (e.g., ~/.local/share/hark)
        """
        self.base_path = base_path
        self.profiles_dir = base_path / "voice-profiles"

    async def create(self, profile: StorageVoiceProfile) -> StorageVoiceProfile:
        """Create a new voice profile."""
        # Ensure directory exists
        await asyncio.to_thread(self.profiles_dir.mkdir, parents=True, exist_ok=True)

        profile_file = self.profiles_dir / f"{profile.id}.json"

        if await asyncio.to_thread(profile_file.exists):
            raise StorageError(
                f"Voice profile {profile.id} already exists",
                details={"profile_id": profile.id},
            )

        try:
            await self._write_profile(profile_file, profile)
            return profile
        except Exception as e:
            # Clean up on failure
            await asyncio.to_thread(profile_file.unlink, missing_ok=True)
            raise StorageError(
                f"Failed to create voice profile: {e}",
                details={"profile_id": profile.id},
            ) from e

    async def get(self, profile_id: str) -> StorageVoiceProfile | None:
        """Get a voice profile by ID."""
        profile_file = self.profiles_dir / f"{profile_id}.json"

        if not await asyncio.to_thread(profile_file.exists):
            return None

        try:
            return await self._read_profile(profile_file)
        except Exception as e:
            raise StorageError(
                f"Failed to read voice profile {profile_id}: {e}",
                details={"profile_id": profile_id},
            ) from e

    async def list(self) -> list[StorageVoiceProfile]:
        """List all voice profiles."""
        try:
            if not await asyncio.to_thread(self.profiles_dir.exists):
                return []

            # Get all JSON files
            profile_files = await asyncio.to_thread(lambda: list(self.profiles_dir.glob("*.json")))

            profiles = []
            for profile_file in profile_files:
                try:
                    profile = await self._read_profile(profile_file)
                    profiles.append(profile)
                except Exception:
                    logger.warning(f"Skipping corrupted voice profile: {profile_file}")
                    continue

            # Sort by name
            profiles.sort(key=lambda p: p.name.lower())

            return profiles

        except Exception as e:
            raise StorageError(f"Failed to list voice profiles: {e}") from e

    async def update(self, profile: StorageVoiceProfile) -> StorageVoiceProfile:
        """Update an existing voice profile."""
        profile_file = self.profiles_dir / f"{profile.id}.json"

        if not await asyncio.to_thread(profile_file.exists):
            raise VoiceProfileNotFoundError(profile.id)

        try:
            await self._write_profile(profile_file, profile)
            return profile
        except Exception as e:
            raise StorageError(
                f"Failed to update voice profile {profile.id}: {e}",
                details={"profile_id": profile.id},
            ) from e

    async def delete(self, profile_id: str) -> None:
        """Delete a voice profile."""
        profile_file = self.profiles_dir / f"{profile_id}.json"

        if not await asyncio.to_thread(profile_file.exists):
            raise VoiceProfileNotFoundError(profile_id)

        try:
            await asyncio.to_thread(profile_file.unlink)
        except Exception as e:
            raise StorageError(
                f"Failed to delete voice profile {profile_id}: {e}",
                details={"profile_id": profile_id},
            ) from e

    # Helper methods

    async def _write_profile(self, profile_file: Path, profile: StorageVoiceProfile) -> None:
        """Write voice profile to JSON file atomically.

        Uses write-to-temp-then-rename pattern for crash safety.
        """
        data = profile.model_dump(mode="json")

        def write_atomic():
            parent_dir = str(profile_file.parent)
            fd, tmp_path = tempfile.mkstemp(dir=parent_dir, suffix=".tmp", prefix=".profile-")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, str(profile_file))
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise

        await asyncio.to_thread(write_atomic)

    async def _read_profile(self, profile_file: Path) -> StorageVoiceProfile:
        """Read voice profile from JSON file."""

        def read():
            with open(profile_file, encoding="utf-8") as f:
                return json.load(f)

        data = await asyncio.to_thread(read)
        return StorageVoiceProfile.model_validate(data)
