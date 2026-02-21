"""Configuration management for harkd using pydantic-settings."""

import functools
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "ServerSettings",
    "CorsSettings",
    "StorageSettings",
    "LoggingSettings",
    "RecordingDefaults",
    "LLMSettings",
    "KoyebProviderSettings",
    "VerdaProviderSettings",
    "ScalewayProviderSettings",
    "FileStorageSettings",
    "InfraSettings",
    "KoyebInfraSettings",
    "ScalewayInfraSettings",
    "DataCrunchInfraSettings",
    "TranscriptionSettings",
    "HarkdSettings",
    "get_settings",
]


class ServerSettings(BaseSettings):
    """Server configuration."""

    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8765, ge=1024, le=65535, description="Server port")
    reload: bool = Field(default=False, description="Auto-reload on code changes")


class CorsSettings(BaseSettings):
    """CORS configuration."""

    enabled: bool = True
    origins: list[str] = Field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    allow_methods: list[str] = Field(default_factory=lambda: ["*"])
    allow_headers: list[str] = Field(default_factory=lambda: ["*"])


class StorageSettings(BaseSettings):
    """Storage configuration."""

    base_path: Path = Field(
        default_factory=lambda: Path.home() / ".local/share/hark",
        description="Base storage path",
    )

    @field_validator("base_path")
    @classmethod
    def ensure_absolute(cls, v: Path) -> Path:
        """Ensure path is absolute and expanded."""
        return v.expanduser().absolute()


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    )
    file: Path | None = None
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class RecordingDefaults(BaseModel):
    """Default recording settings.

    Heavy settings (model) are daemon-only and require restart to change.
    Light settings can be overridden per-recording via the API.
    """

    model: str = Field(default="large-v3", description="Whisper model name or HF model path")
    word_timestamps: bool = Field(default=False, description="Include word-level timestamps")
    language: str = Field(default="auto", description="Language code or 'auto'")
    diarization: bool = Field(default=True, description="Enable speaker diarization")
    noise_reduction: bool = Field(default=True, description="Enable noise reduction")
    normalization: bool = Field(default=True, description="Enable audio normalization")
    mic_gain: float = Field(
        default=2.0, ge=0.1, le=10.0, description="Microphone gain multiplier (1.0 = no gain)"
    )
    beam_size: int = Field(default=3, ge=1, le=10, description="Beam size for decoding (1=greedy)")
    batch_size: int = Field(default=16, ge=1, le=128, description="Batch size for transcription")
    vad_onset: float = Field(default=0.5, ge=0.0, le=1.0, description="VAD onset threshold")
    vad_offset: float = Field(default=0.363, ge=0.0, le=1.0, description="VAD offset threshold")
    vad_method: Literal["pyannote", "silero"] = Field(default="pyannote", description="VAD method")
    transcription_timeout: int = Field(
        default=1800, ge=60, description="Transcription subprocess timeout in seconds"
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Max retry attempts for failed recordings"
    )
    speaker_match_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for matching speakers to voice profiles",
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate Whisper model name or HuggingFace model path."""
        valid_short = [
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
            "distil-large-v3",
            "large-v3-turbo",
        ]
        # Accept short names or HF-style paths (org/model-name)
        if v in valid_short or "/" in v:
            return v
        raise ValueError(
            f"Invalid model. Must be one of {valid_short} "
            "or a HuggingFace model path (e.g. org/model-name)"
        )


class KoyebProviderSettings(BaseModel):
    """Koyeb provider configuration."""

    token: str = Field(description="Koyeb API/Bearer token")


class VerdaProviderSettings(BaseModel):
    """Verda (DataCrunch) provider configuration."""

    api_key: str = Field(description="Verda inference API key")
    poll_interval: int = Field(default=5, ge=1, le=60, description="Async poll interval (seconds)")


class ScalewayProviderSettings(BaseModel):
    """Scaleway provider configuration."""

    api_key: str | None = Field(default=None, description="Auth token for worker endpoint")


class FileStorageSettings(BaseModel):
    """S3-compatible storage for URL-based file transfer (e.g. Verda)."""

    endpoint: str = Field(description="S3 endpoint URL")
    bucket: str = Field(description="Bucket name")
    access_key: str = Field(description="S3 access key")
    secret_key: str = Field(description="S3 secret key")
    region: str = Field(default="auto", description="S3 region")


class InfraSettings(BaseModel):
    """Common infrastructure lifecycle settings."""

    docker_image: str = Field(
        default="ghcr.io/harkhq/harkd/worker:latest",
        description="Worker container image",
    )
    idle_timeout: int = Field(
        default=300, ge=0, description="Seconds of inactivity before teardown (0=never)"
    )
    max_runtime: int = Field(
        default=0, ge=0, description="Max seconds before forced teardown (0=unlimited)"
    )
    provisioning_timeout: int = Field(default=600, ge=60, description="Max wait for provisioning")


class KoyebInfraSettings(BaseModel):
    """Koyeb infrastructure management settings."""

    api_token: str = Field(description="Koyeb API token for service management")
    region: str = Field(default="fra", description="Deployment region")
    instance_type: str = Field(
        default="gpu-nvidia-rtx-4000-sff-ada", description="GPU instance type"
    )
    app_name: str = Field(default="hark-worker", description="Koyeb app/service name")
    use_native_scale_to_zero: bool = Field(
        default=True, description="Let Koyeb handle idle scaling (recommended)"
    )


class ScalewayInfraSettings(BaseModel):
    """Scaleway infrastructure management settings."""

    secret_key: str = Field(description="Scaleway secret key (SCW_SECRET_KEY)")
    organization_id: str = Field(description="Scaleway organization ID")
    project_id: str = Field(description="Scaleway project ID")
    zone: str = Field(default="fr-par-2", description="Availability zone (must support GPUs)")
    instance_type: str = Field(default="L4-1-24G", description="Instance type")
    image_id: str | None = Field(
        default=None, description="Pre-baked OS image UUID (optional, uses cloud-init if unset)"
    )


class DataCrunchInfraSettings(BaseModel):
    """DataCrunch infrastructure management settings."""

    client_id: str = Field(description="DataCrunch OAuth2 client ID")
    client_secret: str = Field(description="DataCrunch OAuth2 client secret")
    instance_type: str = Field(default="1L40S.6V", description="GPU instance type")
    location: str = Field(default="FIN-01", description="Data center location")
    ssh_key_ids: list[str] = Field(default_factory=list, description="SSH key IDs")
    os_volume_id: str | None = Field(default=None, description="Pre-built OS volume ID")


class TranscriptionSettings(BaseModel):
    """Transcription backend configuration."""

    backend: Literal["local", "remote", "koyeb", "verda", "scaleway"] = Field(
        default="local", description="Transcription backend"
    )
    endpoint_url: str | None = Field(default=None, description="Remote worker endpoint URL")
    worker_api_key: str | None = Field(default=None, description="Shared secret for worker auth")
    remote_timeout: int = Field(default=3600, ge=60, description="Remote API timeout (seconds)")
    max_retries: int = Field(default=2, ge=0, le=5, description="Max retries for remote failures")
    fallback_to_local: bool = Field(
        default=True, description="Fall back to local on remote failure"
    )

    # Provider-specific
    koyeb: KoyebProviderSettings | None = None
    verda: VerdaProviderSettings | None = None
    scaleway: ScalewayProviderSettings | None = None

    # Optional S3 for URL-based file transfer
    file_storage: FileStorageSettings | None = None

    # Infrastructure lifecycle management
    managed: bool = Field(default=False, description="Enable infrastructure lifecycle management")
    infra: InfraSettings | None = Field(
        default=None, description="Infrastructure lifecycle settings (required when managed=True)"
    )
    koyeb_infra: KoyebInfraSettings | None = None
    scaleway_infra: ScalewayInfraSettings | None = None
    datacrunch_infra: DataCrunchInfraSettings | None = None

    @model_validator(mode="after")
    def validate_managed_config(self) -> "TranscriptionSettings":
        """Validate managed mode has required infrastructure config."""
        if not self.managed:
            return self

        if self.backend in ("local", "remote"):
            return self

        if self.infra is None:
            raise ValueError("transcription.infra is required when managed=True")

        if self.backend == "koyeb" and self.koyeb_infra is None:
            raise ValueError("transcription.koyeb_infra is required for managed koyeb backend")

        if self.backend == "scaleway" and self.scaleway_infra is None:
            raise ValueError(
                "transcription.scaleway_infra is required for managed scaleway backend"
            )

        if self.backend == "verda" and self.datacrunch_infra is None:
            raise ValueError("transcription.datacrunch_infra is required for managed verda backend")

        return self


class LLMSettings(BaseModel):
    """LLM configuration for AI features (meeting minutes, etc.)."""

    enabled: bool = Field(default=False, description="Enable LLM features (meeting minutes, etc.)")
    provider: Literal["openai", "anthropic", "google", "ollama"] = Field(
        default="openai", description="LLM provider"
    )
    model: str = Field(default="gpt-4o-mini", description="Model name")
    api_key: str | None = Field(default=None, description="API key (not needed for Ollama)")
    base_url: str | None = Field(default=None, description="Custom base URL (e.g. Ollama endpoint)")
    temperature: float = Field(default=0.3, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1)

    # Middleware
    enable_cache: bool = Field(default=False, description="Enable response caching")
    enable_logging: bool = Field(default=True, description="Log LLM requests")
    enable_token_stats: bool = Field(default=True, description="Track token usage")

    # Custom prompts
    custom_prompts_dir: Path | None = Field(
        default=None, description="Directory with custom prompt .txt files"
    )


class HarkdSettings(BaseSettings):
    """
    Main daemon configuration.

    Load order (later overrides earlier):
    1. Default values
    2. Config file (~/.config/hark/daemon.yaml)
    3. Environment variables (HARKD_*)
    4. .env file
    """

    model_config = SettingsConfigDict(
        env_prefix="HARKD_",
        env_file=".env",
        env_nested_delimiter="__",
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    cors: CorsSettings = Field(default_factory=CorsSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    recording: RecordingDefaults = Field(default_factory=RecordingDefaults)
    transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # HuggingFace token for diarization (pyannote models)
    hf_token: str | None = Field(
        default=None,
        description="HuggingFace token for pyannote diarization models",
    )

    @classmethod
    def from_yaml(cls, path: Path) -> "HarkdSettings":
        """Load settings from YAML file."""
        if not path.exists():
            return cls()

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data) if data else cls()


@functools.lru_cache(maxsize=1)
def get_settings() -> HarkdSettings:
    """Get or create settings singleton."""
    config_path = Path.home() / ".config/hark/daemon.yaml"
    return HarkdSettings.from_yaml(config_path)
