"""Tests for harkd configuration."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from harkd.config import (
    CorsSettings,
    HarkdSettings,
    LLMSettings,
    LoggingSettings,
    RecordingDefaults,
    ServerSettings,
    StorageSettings,
    get_settings,
)


def test_server_settings_defaults():
    """Test ServerSettings with defaults."""
    settings = ServerSettings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8765
    assert settings.reload is False


def test_server_settings_custom():
    """Test ServerSettings with custom values."""
    settings = ServerSettings(host="0.0.0.0", port=9000, reload=True)
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.reload is True


def test_server_settings_port_validation():
    """Test ServerSettings port validation."""
    # Port too low
    with pytest.raises(ValueError):
        ServerSettings(port=1023)

    # Port too high
    with pytest.raises(ValueError):
        ServerSettings(port=65536)

    # Valid ports
    ServerSettings(port=1024)  # Min valid
    ServerSettings(port=65535)  # Max valid


def test_cors_settings_defaults():
    """Test CorsSettings with defaults."""
    settings = CorsSettings()
    assert settings.enabled is True
    assert settings.origins == ["*"]
    assert settings.allow_credentials is True


def test_storage_settings_defaults():
    """Test StorageSettings with defaults."""
    settings = StorageSettings()
    expected_path = Path.home() / ".local/share/hark"
    assert settings.base_path == expected_path.expanduser().absolute()


def test_storage_settings_path_expansion():
    """Test StorageSettings expands ~ in paths."""
    settings = StorageSettings(base_path=Path("~/custom/path"))
    assert not str(settings.base_path).startswith("~")
    assert settings.base_path.is_absolute()


def test_logging_settings_defaults():
    """Test LoggingSettings with defaults."""
    settings = LoggingSettings()
    assert settings.level == "INFO"
    assert settings.file is None
    assert "%(asctime)s" in settings.format


def test_logging_settings_level_validation():
    """Test LoggingSettings level validation."""
    # Valid levels
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        settings = LoggingSettings(level=level)
        assert settings.level == level

    # Invalid level
    with pytest.raises(ValueError):
        LoggingSettings(level="INVALID")


def test_recording_defaults():
    """Test RecordingDefaults with defaults."""
    defaults = RecordingDefaults()
    assert defaults.model == "large-v3"
    assert defaults.word_timestamps is False
    assert defaults.language == "auto"
    assert defaults.input_source == "both"
    assert defaults.diarization is True
    assert defaults.noise_reduction is True
    assert defaults.normalization is True


def test_recording_defaults_custom():
    """Test RecordingDefaults with custom values."""
    defaults = RecordingDefaults(
        model="small",
        word_timestamps=True,
        language="en",
        input_source="mic",
        diarization=False,
    )
    assert defaults.model == "small"
    assert defaults.word_timestamps is True
    assert defaults.language == "en"
    assert defaults.input_source == "mic"
    assert defaults.diarization is False


def test_recording_defaults_model_validation():
    """Test RecordingDefaults validates model name."""
    valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    for model in valid_models:
        defaults = RecordingDefaults(model=model)
        assert defaults.model == model

    with pytest.raises(ValueError):
        RecordingDefaults(model="invalid-model")


def test_recording_defaults_input_source_validation():
    """Test RecordingDefaults validates input_source."""
    for source in ["mic", "speaker", "both"]:
        defaults = RecordingDefaults(input_source=source)
        assert defaults.input_source == source

    with pytest.raises(ValueError):
        RecordingDefaults(input_source="invalid")


def test_harkd_settings_defaults():
    """Test HarkdSettings with all defaults."""
    settings = HarkdSettings()
    assert isinstance(settings.server, ServerSettings)
    assert isinstance(settings.cors, CorsSettings)
    assert isinstance(settings.storage, StorageSettings)
    assert isinstance(settings.logging, LoggingSettings)
    assert isinstance(settings.recording, RecordingDefaults)
    assert settings.recording.model == "large-v3"


def test_harkd_settings_nested_config():
    """Test HarkdSettings with nested configuration."""
    settings = HarkdSettings(
        server={"host": "0.0.0.0", "port": 9000},
        logging={"level": "DEBUG"},
    )
    assert settings.server.host == "0.0.0.0"
    assert settings.server.port == 9000
    assert settings.logging.level == "DEBUG"


def test_harkd_settings_from_yaml():
    """Test HarkdSettings loading from YAML file."""
    yaml_content = """
server:
  host: 0.0.0.0
  port: 9000
  reload: true

cors:
  enabled: false
  origins:
    - http://example.com

storage:
  base_path: /tmp/hark-test

logging:
  level: DEBUG
  file: /tmp/harkd.log

recording:
  model: small
  language: en
  input_source: mic
  diarization: false
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = Path(f.name)

    try:
        settings = HarkdSettings.from_yaml(yaml_path)
        assert settings.server.host == "0.0.0.0"
        assert settings.server.port == 9000
        assert settings.server.reload is True
        assert settings.cors.enabled is False
        assert settings.cors.origins == ["http://example.com"]
        assert settings.storage.base_path == Path("/tmp/hark-test").absolute()
        assert settings.logging.level == "DEBUG"
        assert settings.logging.file == Path("/tmp/harkd.log")
        assert settings.recording.model == "small"
        assert settings.recording.language == "en"
        assert settings.recording.input_source == "mic"
        assert settings.recording.diarization is False
        # Unspecified fields should use defaults
        assert settings.recording.word_timestamps is False
        assert settings.recording.noise_reduction is True
    finally:
        yaml_path.unlink()


def test_harkd_settings_from_yaml_nonexistent():
    """Test HarkdSettings with nonexistent YAML file uses defaults."""
    settings = HarkdSettings.from_yaml(Path("/nonexistent/daemon.yaml"))
    # Should return defaults without error
    assert settings.server.host == "127.0.0.1"
    assert settings.server.port == 8765


def test_harkd_settings_from_yaml_empty():
    """Test HarkdSettings with empty YAML file uses defaults."""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")  # Empty file
        yaml_path = Path(f.name)

    try:
        settings = HarkdSettings.from_yaml(yaml_path)
        assert settings.server.host == "127.0.0.1"
        assert settings.server.port == 8765
    finally:
        yaml_path.unlink()


def test_harkd_settings_env_override(monkeypatch):
    """Test HarkdSettings environment variable overrides."""
    monkeypatch.setenv("HARKD_SERVER__HOST", "10.0.0.1")
    monkeypatch.setenv("HARKD_SERVER__PORT", "7000")
    monkeypatch.setenv("HARKD_LOGGING__LEVEL", "ERROR")

    settings = HarkdSettings()
    assert settings.server.host == "10.0.0.1"
    assert settings.server.port == 7000
    assert settings.logging.level == "ERROR"


def test_harkd_settings_recording_env_override(monkeypatch):
    """Test RecordingDefaults can be overridden via environment variables."""
    monkeypatch.setenv("HARKD_RECORDING__MODEL", "small")
    monkeypatch.setenv("HARKD_RECORDING__DIARIZATION", "false")
    monkeypatch.setenv("HARKD_RECORDING__LANGUAGE", "de")

    settings = HarkdSettings()
    assert settings.recording.model == "small"
    assert settings.recording.diarization is False
    assert settings.recording.language == "de"
    # Unset fields should use defaults
    assert settings.recording.input_source == "both"


def test_get_settings_singleton():
    """Test get_settings returns singleton."""
    # Clear any existing singleton
    get_settings.cache_clear()

    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2

    # Clean up
    get_settings.cache_clear()


def test_get_settings_loads_from_default_path(monkeypatch, tmp_path):
    """Test get_settings loads from default config path."""
    get_settings.cache_clear()

    # Create a temporary config directory
    config_dir = tmp_path / ".config" / "hark"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "daemon.yaml"

    yaml_content = """
server:
  port: 7777
"""
    config_file.write_text(yaml_content)

    # Mock Path.home() to return tmp_path
    def mock_home():
        return tmp_path

    monkeypatch.setattr(Path, "home", mock_home)

    settings = get_settings()
    assert settings.server.port == 7777

    # Clean up singleton
    get_settings.cache_clear()


# --- LLMSettings tests ---


def test_llm_settings_defaults():
    """Test LLMSettings with defaults."""
    settings = LLMSettings()
    assert settings.enabled is False
    assert settings.provider == "openai"
    assert settings.model == "gpt-4o-mini"
    assert settings.api_key is None
    assert settings.base_url is None
    assert settings.temperature == 0.3
    assert settings.max_tokens == 4096
    assert settings.enable_cache is False
    assert settings.enable_logging is True
    assert settings.enable_token_stats is True
    assert settings.custom_prompts_dir is None


def test_llm_settings_custom():
    """Test LLMSettings with custom values."""
    settings = LLMSettings(
        enabled=True,
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        api_key="sk-ant-test",
        temperature=0.7,
        max_tokens=8192,
    )
    assert settings.enabled is True
    assert settings.provider == "anthropic"
    assert settings.model == "claude-sonnet-4-5-20250929"
    assert settings.api_key == "sk-ant-test"
    assert settings.temperature == 0.7
    assert settings.max_tokens == 8192


def test_harkd_settings_includes_llm():
    """Test HarkdSettings has llm field with defaults."""
    settings = HarkdSettings()
    assert isinstance(settings.llm, LLMSettings)
    assert settings.llm.enabled is False
    assert settings.llm.provider == "openai"


def test_harkd_settings_llm_env_override(monkeypatch):
    """Test LLMSettings can be overridden via environment variables."""
    monkeypatch.setenv("HARKD_LLM__ENABLED", "true")
    monkeypatch.setenv("HARKD_LLM__PROVIDER", "anthropic")
    monkeypatch.setenv("HARKD_LLM__MODEL", "claude-sonnet-4-5-20250929")
    monkeypatch.setenv("HARKD_LLM__API_KEY", "sk-ant-test")

    settings = HarkdSettings()
    assert settings.llm.enabled is True
    assert settings.llm.provider == "anthropic"
    assert settings.llm.model == "claude-sonnet-4-5-20250929"
    assert settings.llm.api_key == "sk-ant-test"


def test_harkd_settings_llm_from_yaml():
    """Test LLMSettings loading from YAML file."""
    yaml_content = """
recording:
  diarization: false
llm:
  enabled: true
  provider: ollama
  model: llama3.2
  base_url: http://localhost:11434
  temperature: 0.5
  max_tokens: 2048
  enable_cache: true
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = Path(f.name)

    try:
        settings = HarkdSettings.from_yaml(yaml_path)
        assert settings.llm.enabled is True
        assert settings.llm.provider == "ollama"
        assert settings.llm.model == "llama3.2"
        assert settings.llm.base_url == "http://localhost:11434"
        assert settings.llm.temperature == 0.5
        assert settings.llm.max_tokens == 2048
        assert settings.llm.enable_cache is True
        # Defaults for unset fields
        assert settings.llm.api_key is None
        assert settings.llm.enable_logging is True
    finally:
        yaml_path.unlink()
