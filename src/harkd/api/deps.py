"""Dependency injection for FastAPI routes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends

from harkd.config import HarkdSettings, get_settings
from harkd.services.processing_worker import ProcessingWorker
from harkd.services.recording_service import RecordingService
from harkd.services.voice_profile_service import VoiceProfileService
from harkd.storage.filesystem.recordings import FilesystemRecordingStorage
from harkd.storage.filesystem.voice_profiles import FilesystemVoiceProfileStorage

if TYPE_CHECKING:
    from harkd.chat.service import ChatService

__all__ = [
    "get_recording_service",
    "get_voice_profile_service",
    "get_chat_service",
    "get_processing_worker_eager",
    "RecordingServiceDep",
    "VoiceProfileServiceDep",
    "ChatServiceDep",
]


# Service instances cached per storage path
_recording_services: dict[Path, RecordingService] = {}
_voice_profile_services: dict[Path, VoiceProfileService] = {}
_processing_workers: dict[Path, ProcessingWorker] = {}
_chat_services: dict[Path, ChatService] = {}  # noqa: F821


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


def get_processing_worker_eager(storage_path: Path, config: HarkdSettings) -> ProcessingWorker:
    """Get or create processing worker eagerly (no Depends).

    Called from lifespan to ensure worker exists before requests.

    Args:
        storage_path: Storage base path
        config: Daemon settings

    Returns:
        Processing worker instance
    """
    if storage_path not in _processing_workers:
        storage = FilesystemRecordingStorage(storage_path)
        vp_storage = FilesystemVoiceProfileStorage(storage_path)
        voice_profile_svc = VoiceProfileService(vp_storage)
        _processing_workers[storage_path] = ProcessingWorker(
            storage, config, voice_profile_service=voice_profile_svc
        )
    return _processing_workers[storage_path]


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
        worker = get_processing_worker_eager(storage_path, config)
        voice_profile_svc = get_voice_profile_service(storage_path)
        _recording_services[storage_path] = RecordingService(
            storage, config, worker=worker, voice_profile_service=voice_profile_svc
        )
    return _recording_services[storage_path]


def get_chat_service(
    storage_path: Annotated[Path, Depends(get_storage_path)],
    config: Annotated[HarkdSettings, Depends(get_settings)],
) -> ChatService:
    """Get or create chat service for this storage path.

    Requires llm.enabled = true in configuration.

    Args:
        storage_path: Storage base path
        config: Daemon settings

    Returns:
        Chat service instance

    Raises:
        LLMNotConfiguredError: If LLM is not enabled
    """
    from harkd.chat.service import ChatService
    from harkd.chat.tool_executor import ToolExecutor
    from harkd.exceptions import LLMNotConfiguredError
    from harkd.llm.client import LLMClient
    from harkd.storage.filesystem.chat_threads import FilesystemChatThreadStorage

    if not config.llm.enabled:
        raise LLMNotConfiguredError()

    if storage_path not in _chat_services:
        recording_storage = FilesystemRecordingStorage(storage_path)
        thread_storage = FilesystemChatThreadStorage(storage_path)
        llm_client = LLMClient(config.llm)
        tool_executor = ToolExecutor(recording_storage)
        _chat_services[storage_path] = ChatService(
            thread_storage=thread_storage,
            llm_client=llm_client,
            tool_executor=tool_executor,
        )
    return _chat_services[storage_path]


# Type aliases for dependency injection
RecordingServiceDep = Annotated[RecordingService, Depends(get_recording_service)]
VoiceProfileServiceDep = Annotated[VoiceProfileService, Depends(get_voice_profile_service)]
ChatServiceDep = Annotated["ChatService", Depends(get_chat_service)]
