"""API models for settings."""

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "Settings",
]


class Settings(BaseModel):
    """Application settings (read-only, from daemon configuration)."""

    model: str = Field(description="Whisper model name")
    word_timestamps: bool = Field(description="Include word-level timestamps")
    language: str = Field(description="Language code or 'auto'")
    input_source: Literal["mic", "speaker", "both"] = Field(
        description="Audio input source"
    )
    diarization: bool = Field(description="Enable speaker diarization")
    noise_reduction: bool = Field(description="Enable noise reduction")
    normalization: bool = Field(description="Enable audio normalization")
