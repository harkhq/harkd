"""API models for settings."""

from pydantic import BaseModel, Field

__all__ = [
    "Settings",
]


class Settings(BaseModel):
    """Application settings (read-only, from daemon configuration)."""

    model: str = Field(description="Whisper model name")
    word_timestamps: bool = Field(description="Include word-level timestamps")
    language: str = Field(description="Language code or 'auto'")
    diarization: bool = Field(description="Enable speaker diarization")
    noise_reduction: bool = Field(description="Enable noise reduction")
    normalization: bool = Field(description="Enable audio normalization")
    mic_gain: float = Field(description="Microphone gain multiplier (1.0 = no gain)")
