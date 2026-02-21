"""Voice profile service for managing speaker profiles."""

import logging
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np

from harkd.api.models.voice_profile import (
    VoiceProfile,
    VoiceProfileCreate,
    VoiceProfileDetail,
    VoiceProfileListResponse,
)
from harkd.events import Event, get_event_bus
from harkd.exceptions import VoiceProfileNotFoundError
from harkd.storage.base import VoiceProfileStorage
from harkd.storage.models import StorageVoiceProfile

logger = logging.getLogger(__name__)

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
        get_event_bus().emit(Event("invalidate", {"entity": "voice_profiles", "id": profile_id}))

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

    async def get_known_embeddings(self) -> dict[str, tuple[list[float], str]]:
        """Get average embedding + profile ID for all profiles with embeddings.

        Returns:
            Dict mapping profile name to (average_embedding_vector, profile_id).
            Profiles with no embeddings are excluded.
        """
        storage_profiles = await self.storage.list()
        result: dict[str, tuple[list[float], str]] = {}
        for profile in storage_profiles:
            vectors = [e["vector"] for e in profile.embeddings if e.get("vector")]
            if not vectors:
                continue
            avg = np.mean(vectors, axis=0).tolist()
            result[profile.name] = (avg, profile.id)
        return result

    async def find_or_create_by_name(self, name: str) -> StorageVoiceProfile:
        """Find an existing profile by name (case-insensitive) or create a new one.

        Args:
            name: Speaker name to look up or create

        Returns:
            Existing or newly created storage voice profile
        """
        all_profiles = await self.storage.list()
        name_lower = name.lower()
        for profile in all_profiles:
            if profile.name.lower() == name_lower:
                return profile

        # Create new profile
        profile_id = f"profile-{uuid4().hex[:8]}"
        now = datetime.now(UTC)
        storage_profile = StorageVoiceProfile(
            id=profile_id,
            name=name,
            created_at=now,
        )
        created = await self.storage.create(storage_profile)
        logger.info(f"Created voice profile '{name}' ({profile_id})")
        return created

    async def add_embedding(
        self,
        profile_id: str,
        recording_id: str,
        speaker_label: str,
        vector: list[float],
        audio_duration: float,
    ) -> None:
        """Append an embedding to a voice profile.

        Args:
            profile_id: Voice profile ID
            recording_id: Source recording ID
            speaker_label: Speaker label in the recording
            vector: Embedding vector
            audio_duration: Duration of the recording in seconds
        """
        profile = await self.storage.get(profile_id)
        if profile is None:
            raise VoiceProfileNotFoundError(profile_id)

        now = datetime.now(UTC)
        profile.embeddings.append(
            {
                "recording_id": recording_id,
                "speaker_id": speaker_label,
                "timestamp": now.isoformat(),
                "vector": vector,
                "audio_duration": audio_duration,
            }
        )
        profile.clips += 1
        profile.total_seconds += audio_duration
        profile.last_used = now
        # Simple confidence heuristic: more clips = higher confidence, capped at 1.0
        profile.confidence = min(1.0, profile.clips * 0.2)
        await self.storage.update(profile)
        bus = get_event_bus()
        bus.emit(Event("invalidate", {"entity": "voice_profiles", "id": profile_id}))
        bus.emit(Event("invalidate", {"entity": "unassigned_speakers"}))

    async def remove_embedding(
        self, profile_id: str, recording_id: str, speaker_label: str
    ) -> None:
        """Remove an embedding from a voice profile.

        Args:
            profile_id: Voice profile ID
            recording_id: Source recording ID
            speaker_label: Speaker label in the recording

        Raises:
            VoiceProfileNotFoundError: If profile doesn't exist
        """
        profile = await self.storage.get(profile_id)
        if profile is None:
            raise VoiceProfileNotFoundError(profile_id)

        kept = []
        removed_seconds = 0.0
        removed_count = 0
        for e in profile.embeddings:
            if e.get("recording_id") == recording_id and e.get("speaker_id") == speaker_label:
                removed_seconds += e.get("audio_duration", 0.0)
                removed_count += 1
            else:
                kept.append(e)

        if removed_count > 0:
            profile.embeddings = kept
            profile.clips = max(0, profile.clips - removed_count)
            profile.total_seconds = max(0.0, profile.total_seconds - removed_seconds)
            profile.confidence = min(1.0, profile.clips * 0.2)
            await self.storage.update(profile)

    async def delete_profile(self, profile_id: str) -> None:
        """Delete a voice profile.

        Args:
            profile_id: Profile ID

        Raises:
            VoiceProfileNotFoundError: If profile doesn't exist
        """
        await self.storage.delete(profile_id)
        bus = get_event_bus()
        bus.emit(Event("invalidate", {"entity": "voice_profiles", "id": profile_id}))
        bus.emit(Event("invalidate", {"entity": "unassigned_speakers"}))

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
                    vector=emb_dict.get("vector", []),
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
