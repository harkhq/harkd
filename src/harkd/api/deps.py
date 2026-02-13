"""Dependency injection for FastAPI routes."""

from pathlib import Path
from typing import Annotated

from fastapi import Depends

from harkd.config import HarkdSettings, get_settings
from harkd.services.recording_service import RecordingService
from harkd.services.voice_profile_service import VoiceProfileService
from harkd.storage.filesystem.recordings import FilesystemRecordingStorage
from harkd.storage.filesystem.voice_profiles import FilesystemVoiceProfileStorage

__all__ = [
    "get_recording_service",
    "get_voice_profile_service",
    "RecordingServiceDep",
    "VoiceProfileServiceDep",
]


# Service instances cached per storage path
_recording_services: dict[Path, RecordingService] = {}
_voice_profile_services: dict[Path, VoiceProfileService] = {}


def get_storage_path(settings: Annotated[HarkdSettings, Depends(get_settings)]) -> Path:
    """Get storage base path from settings.

    Args:
        settings: Daemon settings

    Returns:
        Storage base path
    """
    return settings.storage.base_path


def get_voice_profile_service(
    storage_path: Annotated[Path, Depends(get_storage_path)],
) -> VoiceProfileService:
    """Get or create voice profile service for this storage path.

    Args:
        storage_path: Storage base path

    Returns:
        Voice profile service instance
    """
    if storage_path not in _voice_profile_services:
        storage = FilesystemVoiceProfileStorage(storage_path)
        _voice_profile_services[storage_path] = VoiceProfileService(storage)
    return _voice_profile_services[storage_path]


def get_recording_service(
    storage_path: Annotated[Path, Depends(get_storage_path)],
    config: Annotated[HarkdSettings, Depends(get_settings)],
) -> RecordingService:
    """Get or create recording service for this storage path.

    Args:
        storage_path: Storage base path
        config: Daemon settings (provides recording defaults)

    Returns:
        Recording service instance
    """
    if storage_path not in _recording_services:
        storage = FilesystemRecordingStorage(storage_path)
        _recording_services[storage_path] = RecordingService(storage, config)
    return _recording_services[storage_path]


# Type aliases for dependency injection
RecordingServiceDep = Annotated[RecordingService, Depends(get_recording_service)]
VoiceProfileServiceDep = Annotated[VoiceProfileService, Depends(get_voice_profile_service)]
