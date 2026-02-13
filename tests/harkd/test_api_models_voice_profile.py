"""Tests for voice profile API models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from harkd.api.models.voice_profile import (
    VoiceEmbedding,
    VoiceProfile,
    VoiceProfileCreate,
    VoiceProfileDetail,
    VoiceProfileListResponse,
)


class TestVoiceEmbedding:
    """Test VoiceEmbedding model."""

    def test_embedding_creation(self):
        """Test creating voice embedding."""
        timestamp = datetime(2026, 1, 15, 10, 30, 0)
        embedding = VoiceEmbedding(
            recording_id="rec-123", speaker_id="SPEAKER_00", timestamp=timestamp
        )
        assert embedding.recording_id == "rec-123"
        assert embedding.speaker_id == "SPEAKER_00"
        assert embedding.timestamp == timestamp


class TestVoiceProfile:
    """Test VoiceProfile model."""

    def test_profile_minimal(self):
        """Test creating profile with minimal fields."""
        created_at = datetime(2026, 1, 10, 14, 20, 0)
        profile = VoiceProfile(id="prof-001", name="Alice", created_at=created_at)
        assert profile.id == "prof-001"
        assert profile.name == "Alice"
        assert profile.created_at == created_at
        assert profile.last_used is None
        assert profile.clips == 0
        assert profile.total_seconds == 0
        assert profile.confidence == 0

    def test_profile_complete(self):
        """Test creating profile with all fields."""
        created_at = datetime(2026, 1, 10, 14, 20, 0)
        last_used = datetime(2026, 1, 15, 10, 30, 0)
        profile = VoiceProfile(
            id="prof-001",
            name="Alice",
            created_at=created_at,
            last_used=last_used,
            clips=12,
            total_seconds=450.3,
            confidence=0.92,
        )
        assert profile.last_used == last_used
        assert profile.clips == 12
        assert profile.total_seconds == 450.3
        assert profile.confidence == 0.92

    def test_profile_name_validation(self):
        """Test name validation."""
        created_at = datetime.now()

        # Empty name should fail
        with pytest.raises(ValidationError):
            VoiceProfile(id="prof-001", name="", created_at=created_at)

        # Name too long should fail
        long_name = "x" * 101
        with pytest.raises(ValidationError):
            VoiceProfile(id="prof-001", name=long_name, created_at=created_at)

        # Valid name should pass
        profile = VoiceProfile(id="prof-001", name="A", created_at=created_at)
        assert profile.name == "A"

    def test_profile_clips_validation(self):
        """Test clips must be non-negative."""
        created_at = datetime.now()
        with pytest.raises(ValidationError):
            VoiceProfile(id="prof-001", name="Alice", created_at=created_at, clips=-1)

    def test_profile_total_seconds_validation(self):
        """Test total_seconds must be non-negative."""
        created_at = datetime.now()
        with pytest.raises(ValidationError):
            VoiceProfile(
                id="prof-001", name="Alice", created_at=created_at, total_seconds=-1.0
            )

    def test_profile_confidence_validation(self):
        """Test confidence must be 0-1."""
        created_at = datetime.now()

        # Negative confidence should fail
        with pytest.raises(ValidationError):
            VoiceProfile(
                id="prof-001", name="Alice", created_at=created_at, confidence=-0.1
            )

        # Confidence > 1 should fail
        with pytest.raises(ValidationError):
            VoiceProfile(
                id="prof-001", name="Alice", created_at=created_at, confidence=1.1
            )

        # Valid confidence should pass
        profile = VoiceProfile(
            id="prof-001", name="Alice", created_at=created_at, confidence=0.85
        )
        assert profile.confidence == 0.85


class TestVoiceProfileDetail:
    """Test VoiceProfileDetail model."""

    def test_detail_with_embeddings(self):
        """Test profile detail with embeddings."""
        created_at = datetime(2026, 1, 10, 14, 20, 0)
        embeddings = [
            VoiceEmbedding(
                recording_id="rec-1", speaker_id="SPEAKER_00", timestamp=created_at
            ),
            VoiceEmbedding(
                recording_id="rec-2", speaker_id="SPEAKER_01", timestamp=created_at
            ),
        ]
        detail = VoiceProfileDetail(
            id="prof-001",
            name="Alice",
            created_at=created_at,
            clips=2,
            embeddings=embeddings,
        )
        assert len(detail.embeddings) == 2
        assert detail.embeddings[0].recording_id == "rec-1"
        assert detail.embeddings[1].recording_id == "rec-2"

    def test_detail_without_embeddings(self):
        """Test profile detail defaults to empty embeddings list."""
        created_at = datetime.now()
        detail = VoiceProfileDetail(id="prof-001", name="Alice", created_at=created_at)
        assert detail.embeddings == []


class TestVoiceProfileCreate:
    """Test VoiceProfileCreate model."""

    def test_create_valid(self):
        """Test creating profile creation request."""
        create = VoiceProfileCreate(name="Alice")
        assert create.name == "Alice"

    def test_create_name_validation(self):
        """Test name validation."""
        # Empty name should fail
        with pytest.raises(ValidationError):
            VoiceProfileCreate(name="")

        # Name too long should fail
        long_name = "x" * 101
        with pytest.raises(ValidationError):
            VoiceProfileCreate(name=long_name)

        # Valid name should pass
        create = VoiceProfileCreate(name="Alice")
        assert create.name == "Alice"


class TestVoiceProfileListResponse:
    """Test VoiceProfileListResponse model."""

    def test_list_response_empty(self):
        """Test empty list response."""
        response = VoiceProfileListResponse(profiles=[])
        assert response.profiles == []

    def test_list_response_with_profiles(self):
        """Test list response with profiles."""
        created_at = datetime.now()
        profiles = [
            VoiceProfile(id="prof-1", name="Alice", created_at=created_at),
            VoiceProfile(id="prof-2", name="Bob", created_at=created_at),
        ]
        response = VoiceProfileListResponse(profiles=profiles)
        assert len(response.profiles) == 2
        assert response.profiles[0].name == "Alice"
        assert response.profiles[1].name == "Bob"
