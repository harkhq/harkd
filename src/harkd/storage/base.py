"""Abstract base classes for storage layer.

Defines interfaces for storage implementations.
Currently only filesystem, but designed for future extensibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from harkd.storage.models import (
    SpeakerEmbeddings,
    StorageChatThread,
    StorageRecording,
    StorageVoiceProfile,
)

__all__ = [
    "ChatThreadStorage",
    "RecordingStorage",
    "VoiceProfileStorage",
]


class RecordingStorage(ABC):
    """Abstract interface for recording storage."""

    @abstractmethod
    async def create(self, recording: StorageRecording) -> StorageRecording:
        """Create a new recording.

        Args:
            recording: Recording to create

        Returns:
            Created recording

        Raises:
            StorageError: If creation fails
        """
        pass

    @abstractmethod
    async def get(self, recording_id: str) -> StorageRecording | None:
        """Get a recording by ID.

        Args:
            recording_id: Recording ID

        Returns:
            Recording if found, None otherwise

        Raises:
            StorageError: If read fails
        """
        pass

    @abstractmethod
    async def list(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> list[StorageRecording]:
        """List recordings with pagination and filtering.

        Args:
            limit: Maximum number of recordings to return
            offset: Number of recordings to skip
            status: Optional status filter

        Returns:
            List of recordings

        Raises:
            StorageError: If read fails
        """
        pass

    @abstractmethod
    async def update(self, recording: StorageRecording) -> StorageRecording:
        """Update an existing recording.

        Args:
            recording: Recording with updates

        Returns:
            Updated recording

        Raises:
            StorageError: If update fails
            RecordingNotFoundError: If recording doesn't exist
        """
        pass

    @abstractmethod
    async def delete(self, recording_id: str) -> None:
        """Delete a recording.

        Args:
            recording_id: Recording ID

        Raises:
            StorageError: If deletion fails
            RecordingNotFoundError: If recording doesn't exist
        """
        pass

    @abstractmethod
    async def count(self, status: str | None = None) -> int:
        """Count recordings.

        Args:
            status: Optional status filter

        Returns:
            Number of recordings

        Raises:
            StorageError: If count fails
        """
        pass

    @abstractmethod
    async def get_speaker_embeddings(self, recording_id: str) -> SpeakerEmbeddings | None:
        """Get speaker embeddings for a recording.

        Reads only the embeddings file, not the full metadata.

        Args:
            recording_id: Recording ID

        Returns:
            Speaker embeddings dict if available, None otherwise

        Raises:
            StorageError: If read fails
        """
        pass


class ChatThreadStorage(ABC):
    """Abstract interface for chat thread storage."""

    @abstractmethod
    async def create(self, thread: StorageChatThread) -> StorageChatThread:
        """Create a new chat thread."""
        pass

    @abstractmethod
    async def get(self, thread_id: str) -> StorageChatThread | None:
        """Get a chat thread by ID."""
        pass

    @abstractmethod
    async def list(self) -> list[StorageChatThread]:
        """List all chat threads (metadata only, messages stripped)."""
        pass

    @abstractmethod
    async def update(self, thread: StorageChatThread) -> StorageChatThread:
        """Update an existing chat thread."""
        pass

    @abstractmethod
    async def delete(self, thread_id: str) -> None:
        """Delete a chat thread."""
        pass


class VoiceProfileStorage(ABC):
    """Abstract interface for voice profile storage."""

    @abstractmethod
    async def create(self, profile: StorageVoiceProfile) -> StorageVoiceProfile:
        """Create a new voice profile.

        Args:
            profile: Profile to create

        Returns:
            Created profile

        Raises:
            StorageError: If creation fails
        """
        pass

    @abstractmethod
    async def get(self, profile_id: str) -> StorageVoiceProfile | None:
        """Get a voice profile by ID.

        Args:
            profile_id: Profile ID

        Returns:
            Profile if found, None otherwise

        Raises:
            StorageError: If read fails
        """
        pass

    @abstractmethod
    async def list(self) -> list[StorageVoiceProfile]:
        """List all voice profiles.

        Returns:
            List of profiles

        Raises:
            StorageError: If read fails
        """
        pass

    @abstractmethod
    async def update(self, profile: StorageVoiceProfile) -> StorageVoiceProfile:
        """Update an existing voice profile.

        Args:
            profile: Profile with updates

        Returns:
            Updated profile

        Raises:
            StorageError: If update fails
            VoiceProfileNotFoundError: If profile doesn't exist
        """
        pass

    @abstractmethod
    async def delete(self, profile_id: str) -> None:
        """Delete a voice profile.

        Args:
            profile_id: Profile ID

        Raises:
            StorageError: If deletion fails
            VoiceProfileNotFoundError: If profile doesn't exist
        """
        pass
