"""Transcription backend protocol and request type."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TranscriptionRequest:
    """Parameters for a transcription job."""

    audio_path: Path
    model_name: str
    language: str | None
    word_timestamps: bool
    diarize: bool
    hf_token: str | None
    beam_size: int
    batch_size: int
    vad_onset: float
    vad_offset: float
    vad_method: str
    timeout: int
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    clustering_threshold: float | None = None

    def to_params_dict(self) -> dict[str, Any]:
        """Convert to dict suitable for JSON serialization (for remote workers)."""
        params: dict[str, Any] = {
            "model_name": self.model_name,
            "language": self.language,
            "word_timestamps": self.word_timestamps,
            "diarize": self.diarize,
            "beam_size": self.beam_size,
            "batch_size": self.batch_size,
            "vad_onset": self.vad_onset,
            "vad_offset": self.vad_offset,
            "vad_method": self.vad_method,
        }
        if self.hf_token:
            params["hf_token"] = self.hf_token
        if self.num_speakers is not None:
            params["num_speakers"] = self.num_speakers
        if self.min_speakers is not None:
            params["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            params["max_speakers"] = self.max_speakers
        if self.clustering_threshold is not None:
            params["clustering_threshold"] = self.clustering_threshold
        return params


@dataclass
class BackendHealth:
    """Health check result from a transcription backend."""

    healthy: bool
    details: dict[str, Any] = field(default_factory=dict)


class TranscriptionBackend(ABC):
    """Abstract base class for transcription backends.

    The result dict format matches transcribe_audio_worker() output:
    text, language, language_probability, duration, segments, speakers, speaker_embeddings.
    """

    @abstractmethod
    async def transcribe(self, request: TranscriptionRequest) -> dict[str, Any]:
        """Run transcription and return result dict."""
        ...

    @abstractmethod
    async def health_check(self) -> BackendHealth:
        """Check if the backend is available."""
        ...

    async def close(self) -> None:  # noqa: B027
        """Clean up resources. Override in subclasses that hold connections."""
