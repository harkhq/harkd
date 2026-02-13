"""Voice profile API routes."""

import logging

from fastapi import APIRouter, status

from harkd.api.deps import VoiceProfileServiceDep
from harkd.api.models.voice_profile import (
    VoiceProfile,
    VoiceProfileCreate,
    VoiceProfileDetail,
    VoiceProfileListResponse,
)

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice-profiles", tags=["Voice Profiles"])


@router.get("", response_model=VoiceProfileListResponse)
async def list_profiles(service: VoiceProfileServiceDep) -> VoiceProfileListResponse:
    """List all voice profiles.

    Returns:
        List of voice profiles sorted by name
    """
    logger.debug("GET /voice-profiles - listing all profiles")
    result = await service.list_profiles()
    logger.debug(f"Returning {len(result.profiles)} voice profiles")
    return result


@router.get("/{profile_id}", response_model=VoiceProfileDetail)
async def get_profile(
    profile_id: str,
    service: VoiceProfileServiceDep,
) -> VoiceProfileDetail:
    """Get detailed voice profile by ID.

    Args:
        profile_id: Profile ID
        service: Voice profile service

    Returns:
        Detailed voice profile with embeddings

    Raises:
        VoiceProfileNotFoundError: If profile doesn't exist (404)
    """
    logger.debug(f"GET /voice-profiles/{profile_id}")
    return await service.get_profile_detail(profile_id)


@router.post("", response_model=VoiceProfile, status_code=status.HTTP_201_CREATED)
async def create_profile(
    create: VoiceProfileCreate,
    service: VoiceProfileServiceDep,
) -> VoiceProfile:
    """Create a new voice profile.

    Args:
        create: Profile creation request
        service: Voice profile service

    Returns:
        Created voice profile
    """
    logger.info(f"POST /voice-profiles - creating profile: {create.name}")
    result = await service.create_profile(create)
    logger.info(f"Voice profile {result.id} created successfully")
    return result


@router.delete("/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_profile(
    profile_id: str,
    service: VoiceProfileServiceDep,
) -> None:
    """Delete a voice profile.

    Args:
        profile_id: Profile ID
        service: Voice profile service

    Raises:
        VoiceProfileNotFoundError: If profile doesn't exist (404)
    """
    logger.info(f"DELETE /voice-profiles/{profile_id}")
    await service.delete_profile(profile_id)
    logger.info(f"Voice profile {profile_id} deleted successfully")
