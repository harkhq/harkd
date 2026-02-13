"""Voice profile service for managing speaker profiles."""

from datetime import UTC, datetime
from uuid import uuid4

from harkd.api.models.voice_profile import (
    VoiceProfile,
    VoiceProfileCreate,
    VoiceProfileDetail,
    VoiceProfileListResponse,
)
from harkd.exceptions import VoiceProfileNotFoundError
from harkd.storage.base import VoiceProfileStorage
from harkd.storage.models import StorageVoiceProfile

__all__ = ["VoiceProfileService"]


class VoiceProfileService:
    """Service for managing voice profiles."""

    def __init__(self, storage: VoiceProfileStorage):
        """Initialize voice profile service.

        Args:
            storage: Voice profile storage backend
        """
        self.storage = storage

    async def create_profile(self, create: VoiceProfileCreate) -> VoiceProfile:
        """Create a new voice profile.

        Args:
            create: Profile creation request

        Returns:
            Created voice profile
        """
        # Generate unique ID
        profile_id = f"profile-{uuid4().hex[:8]}"
        now = datetime.now(UTC)

        # Create storage model
        storage_profile = StorageVoiceProfile(
            id=profile_id,
            name=create.name,
            created_at=now,
            last_used=None,
            clips=0,
            total_seconds=0,
            confidence=0,
            embeddings=[],
        )

        # Save to storage
        created = await self.storage.create(storage_profile)

        # Convert to API model
        return self._to_api_model(created)

    async def get_profile(self, profile_id: str) -> VoiceProfile:
        """Get a voice profile by ID.

        Args:
            profile_id: Profile ID

        Returns:
            Voice profile

        Raises:
            VoiceProfileNotFoundError: If profile doesn't exist
        """
        storage_profile = await self.storage.get(profile_id)
        if storage_profile is None:
            raise VoiceProfileNotFoundError(profile_id)

        return self._to_api_model(storage_profile)

    async def get_profile_detail(self, profile_id: str) -> VoiceProfileDetail:
        """Get detailed voice profile with embeddings.

        Args:
            profile_id: Profile ID

        Returns:
            Detailed voice profile with embeddings

        Raises:
            VoiceProfileNotFoundError: If profile doesn't exist
        """
        storage_profile = await self.storage.get(profile_id)
        if storage_profile is None:
            raise VoiceProfileNotFoundError(profile_id)

        return self._to_detail_model(storage_profile)

    async def list_profiles(self) -> VoiceProfileListResponse:
        """List all voice profiles.

        Returns:
            List of voice profiles
        """
        storage_profiles = await self.storage.list()
        api_profiles = [self._to_api_model(p) for p in storage_profiles]
        return VoiceProfileListResponse(profiles=api_profiles)

    async def delete_profile(self, profile_id: str) -> None:
        """Delete a voice profile.

        Args:
            profile_id: Profile ID

        Raises:
            VoiceProfileNotFoundError: If profile doesn't exist
        """
        await self.storage.delete(profile_id)

    def _to_api_model(self, storage: StorageVoiceProfile) -> VoiceProfile:
        """Convert storage model to API model.

        Args:
            storage: Storage voice profile

        Returns:
            API voice profile
        """
        return VoiceProfile(
            id=storage.id,
            name=storage.name,
            created_at=storage.created_at,
            last_used=storage.last_used,
            clips=storage.clips,
            total_seconds=storage.total_seconds,
            confidence=storage.confidence,
        )

    def _to_detail_model(self, storage: StorageVoiceProfile) -> VoiceProfileDetail:
        """Convert storage model to detailed API model.

        Args:
            storage: Storage voice profile

        Returns:
            Detailed API voice profile with embeddings
        """
        from harkd.api.models.voice_profile import VoiceEmbedding

        # Convert embeddings
        embeddings = []
        for emb_dict in storage.embeddings:
            embeddings.append(
                VoiceEmbedding(
                    recording_id=emb_dict["recording_id"],
                    speaker_id=emb_dict["speaker_id"],
                    timestamp=emb_dict["timestamp"],
                )
            )

        return VoiceProfileDetail(
            id=storage.id,
            name=storage.name,
            created_at=storage.created_at,
            last_used=storage.last_used,
            clips=storage.clips,
            total_seconds=storage.total_seconds,
            confidence=storage.confidence,
            embeddings=embeddings,
        )
