"""Voice profile API routes."""

import logging

from fastapi import APIRouter, HTTPException, status

from harkd.api.deps import RecordingServiceDep, VoiceProfileServiceDep
from harkd.api.models.voice_profile import (
    ClipAssignment,
    ProfileClip,
    ProfileClipsResponse,
    SpeakerSegment,
    UnassignedSpeaker,
    UnassignedSpeakersResponse,
    VoiceProfile,
    VoiceProfileCreate,
    VoiceProfileDetail,
    VoiceProfileListResponse,
)

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice-profiles", tags=["Voice Profiles"])


def _segments_for_speaker(
    segments: list[dict], speaker_label: str, limit: int = 20
) -> list[SpeakerSegment]:
    """Extract top segments for a speaker sorted by duration descending."""
    matching = [
        SpeakerSegment(
            start=seg["start"],
            end=seg["end"],
            text=seg.get("text", ""),
        )
        for seg in segments
        if seg.get("speaker") == speaker_label
    ]
    matching.sort(key=lambda s: s.end - s.start, reverse=True)
    return matching[:limit]


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


@router.get("/unassigned-speakers", response_model=UnassignedSpeakersResponse)
async def list_unassigned_speakers(
    service: VoiceProfileServiceDep,
    recording_service: RecordingServiceDep,
) -> UnassignedSpeakersResponse:
    """List speakers from recordings that are not assigned to any voice profile.

    Returns:
        List of unassigned speakers with their audio segments
    """
    logger.debug("GET /voice-profiles/unassigned-speakers")

    # Get set of valid profile IDs
    profiles_resp = await service.list_profiles()
    valid_profile_ids = {p.id for p in profiles_resp.profiles}

    # List all complete recordings
    all_recordings = await recording_service.storage.list(limit=0, offset=0, status="complete")

    speakers: list[UnassignedSpeaker] = []

    for rec in all_recordings:
        # Load embeddings separately (not included in list() results)
        embeddings = await recording_service.storage.get_speaker_embeddings(rec.id)
        if not embeddings:
            continue

        # Build set of assigned speakers for this recording
        assigned_labels = set()
        if rec.speaker_profiles:
            for label, profile_id in rec.speaker_profiles.items():
                if profile_id in valid_profile_ids:
                    assigned_labels.add(label)

        for speaker_label in embeddings:
            if speaker_label in assigned_labels:
                continue

            segments = _segments_for_speaker(rec.segments, speaker_label)
            if not segments:
                continue

            speakers.append(
                UnassignedSpeaker(
                    recording_id=rec.id,
                    recording_title=rec.title,
                    recording_created_at=rec.created_at,
                    speaker_label=speaker_label,
                    segments=segments,
                )
            )

    # Sort by recording date descending
    speakers.sort(key=lambda s: s.recording_created_at, reverse=True)

    logger.debug(f"Found {len(speakers)} unassigned speakers")
    return UnassignedSpeakersResponse(speakers=speakers)


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


@router.get("/{profile_id}/clips", response_model=ProfileClipsResponse)
async def get_profile_clips(
    profile_id: str,
    service: VoiceProfileServiceDep,
    recording_service: RecordingServiceDep,
) -> ProfileClipsResponse:
    """Get audio clips associated with a voice profile.

    Args:
        profile_id: Profile ID

    Returns:
        Profile clips with audio segments
    """
    logger.debug(f"GET /voice-profiles/{profile_id}/clips")

    detail = await service.get_profile_detail(profile_id)

    # Extract unique (recording_id, speaker_id) pairs
    seen = set()
    clips: list[ProfileClip] = []

    for emb in detail.embeddings:
        key = (emb.recording_id, emb.speaker_id)
        if key in seen:
            continue
        seen.add(key)

        # Load recording
        rec = await recording_service.storage.get(emb.recording_id)
        if rec is None:
            continue

        segments = _segments_for_speaker(rec.segments, emb.speaker_id)
        clips.append(
            ProfileClip(
                recording_id=rec.id,
                recording_title=rec.title,
                recording_created_at=rec.created_at,
                speaker_label=emb.speaker_id,
                segments=segments,
            )
        )

    return ProfileClipsResponse(
        profile_id=detail.id,
        profile_name=detail.name,
        clips=clips,
    )


@router.post(
    "/{profile_id}/clips",
    response_model=VoiceProfile,
    status_code=status.HTTP_201_CREATED,
)
async def assign_clip(
    profile_id: str,
    assignment: ClipAssignment,
    service: VoiceProfileServiceDep,
    recording_service: RecordingServiceDep,
) -> VoiceProfile:
    """Assign a speaker clip to a voice profile.

    Args:
        profile_id: Profile ID
        assignment: Clip assignment request

    Returns:
        Updated voice profile
    """
    logger.info(
        f"POST /voice-profiles/{profile_id}/clips - "
        f"recording={assignment.recording_id}, speaker={assignment.speaker_label}"
    )

    # Validate profile exists
    await service.get_profile(profile_id)

    # Load recording and get embedding
    rec = await recording_service.storage.get(assignment.recording_id)
    if rec is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": {
                    "code": "RECORDING_NOT_FOUND",
                    "message": f"Recording not found: {assignment.recording_id}",
                }
            },
        )

    if not rec.speaker_embeddings or assignment.speaker_label not in rec.speaker_embeddings:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": {
                    "code": "EMBEDDING_NOT_FOUND",
                    "message": (
                        f"No embedding for speaker '{assignment.speaker_label}' in recording"
                    ),
                }
            },
        )

    # Check for existing assignment
    existing_profile_id = (rec.speaker_profiles or {}).get(assignment.speaker_label)
    if existing_profile_id == profile_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": {
                    "code": "ALREADY_ASSIGNED",
                    "message": "Speaker is already assigned to this profile",
                }
            },
        )

    # Remove from old profile if reassigning
    if existing_profile_id:
        try:
            await service.remove_embedding(
                existing_profile_id,
                assignment.recording_id,
                assignment.speaker_label,
            )
        except Exception:
            logger.warning(f"Failed to remove old embedding from profile {existing_profile_id}")

    vector = rec.speaker_embeddings[assignment.speaker_label]

    # Calculate audio duration for the speaker
    audio_duration = sum(
        seg["end"] - seg["start"]
        for seg in rec.segments
        if seg.get("speaker") == assignment.speaker_label
    )

    await service.add_embedding(
        profile_id=profile_id,
        recording_id=assignment.recording_id,
        speaker_label=assignment.speaker_label,
        vector=vector,
        audio_duration=audio_duration,
    )

    # Update recording's speaker_profiles
    if rec.speaker_profiles is None:
        rec.speaker_profiles = {}
    rec.speaker_profiles[assignment.speaker_label] = profile_id
    await recording_service.storage.update(rec)

    return await service.get_profile(profile_id)


@router.delete(
    "/{profile_id}/clips/{recording_id}/{speaker_label}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_clip(
    profile_id: str,
    recording_id: str,
    speaker_label: str,
    service: VoiceProfileServiceDep,
    recording_service: RecordingServiceDep,
) -> None:
    """Remove a clip from a voice profile.

    Args:
        profile_id: Profile ID
        recording_id: Recording ID
        speaker_label: Speaker label
    """
    logger.info(f"DELETE /voice-profiles/{profile_id}/clips/{recording_id}/{speaker_label}")

    await service.remove_embedding(profile_id, recording_id, speaker_label)

    # Update recording's speaker_profiles
    rec = await recording_service.storage.get(recording_id)
    if rec is not None and rec.speaker_profiles and speaker_label in rec.speaker_profiles:
        del rec.speaker_profiles[speaker_label]
        await recording_service.storage.update(rec)


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
