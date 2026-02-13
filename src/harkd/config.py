"""Configuration management for harkd using pydantic-settings."""

import functools
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "ServerSettings",
    "CorsSettings",
    "StorageSettings",
    "LoggingSettings",
    "RecordingDefaults",
    "LLMSettings",
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
    origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",  # Vite dev
            "http://localhost:3000",  # Common dev port
            "moz-extension://*",  # Firefox extension
        ]
    )
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

    model: str = Field(default="large-v3", description="Whisper model name")
    word_timestamps: bool = Field(
        default=False, description="Include word-level timestamps"
    )
    language: str = Field(default="auto", description="Language code or 'auto'")
    input_source: Literal["mic", "speaker", "both"] = Field(
        default="both", description="Audio input source"
    )
    diarization: bool = Field(default=True, description="Enable speaker diarization")
    noise_reduction: bool = Field(default=True, description="Enable noise reduction")
    normalization: bool = Field(default=True, description="Enable audio normalization")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate Whisper model name."""
        valid = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if v not in valid:
            raise ValueError(f"Invalid model. Must be one of: {valid}")
        return v


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
