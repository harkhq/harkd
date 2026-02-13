"""Settings API routes (read-only)."""

from typing import Annotated

from fastapi import APIRouter, Depends

from harkd.api.models.settings import Settings
from harkd.config import HarkdSettings, get_settings

__all__ = ["router"]

router = APIRouter(prefix="/settings", tags=["Settings"])


@router.get("", response_model=Settings)
async def get_current_settings(
    config: Annotated[HarkdSettings, Depends(get_settings)],
) -> Settings:
    """Get current daemon recording defaults (read-only).

    Returns recording configuration from daemon.yaml / environment variables.
    To change defaults, edit daemon.yaml and restart the daemon.
    """
    return Settings(**config.recording.model_dump())
