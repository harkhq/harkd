"""Tests for harkd configuration."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from pydantic import ValidationError

from harkd.config import (
    CorsSettings,
    DataCrunchInfraSettings,
    FileStorageSettings,
    HarkdSettings,
    InfraSettings,
    KoyebInfraSettings,
    KoyebProviderSettings,
    LLMSettings,
    LoggingSettings,
    RecordingDefaults,
    ScalewayInfraSettings,
    ServerSettings,
    StorageSettings,
    TranscriptionSettings,
    VerdaProviderSettings,
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
    assert defaults.diarization is True
    assert defaults.noise_reduction is True
    assert defaults.normalization is True
    assert defaults.mic_gain == 2.0


def test_recording_defaults_custom():
    """Test RecordingDefaults with custom values."""
    defaults = RecordingDefaults(
        model="small",
        word_timestamps=True,
        language="en",
        diarization=False,
    )
    assert defaults.model == "small"
    assert defaults.word_timestamps is True
    assert defaults.language == "en"
    assert defaults.diarization is False


def test_recording_defaults_model_validation():
    """Test RecordingDefaults validates model name."""
    valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    for model in valid_models:
        defaults = RecordingDefaults(model=model)
        assert defaults.model == model

    with pytest.raises(ValueError):
        RecordingDefaults(model="invalid-model")


def test_recording_defaults_speaker_match_threshold():
    """Test RecordingDefaults speaker_match_threshold field."""
    defaults = RecordingDefaults()
    assert defaults.speaker_match_threshold == 0.7

    # Custom value
    defaults = RecordingDefaults(speaker_match_threshold=0.5)
    assert defaults.speaker_match_threshold == 0.5

    # Validation: out of range
    with pytest.raises(ValidationError):
        RecordingDefaults(speaker_match_threshold=-0.1)
    with pytest.raises(ValidationError):
        RecordingDefaults(speaker_match_threshold=1.1)


def test_recording_defaults_mic_gain_validation():
    """Test RecordingDefaults mic_gain field validation."""
    # Custom value
    defaults = RecordingDefaults(mic_gain=5.0)
    assert defaults.mic_gain == 5.0

    # Validation: out of range
    with pytest.raises(ValidationError):
        RecordingDefaults(mic_gain=0.05)
    with pytest.raises(ValidationError):
        RecordingDefaults(mic_gain=10.1)


def test_recording_defaults_no_input_source():
    """Test RecordingDefaults no longer has input_source field."""
    assert "input_source" not in RecordingDefaults.model_fields


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


# --- TranscriptionSettings tests ---


def test_transcription_settings_defaults():
    """Test TranscriptionSettings with defaults."""
    settings = TranscriptionSettings()
    assert settings.backend == "local"
    assert settings.endpoint_url is None
    assert settings.worker_api_key is None
    assert settings.remote_timeout == 3600
    assert settings.max_retries == 2
    assert settings.fallback_to_local is True
    assert settings.koyeb is None
    assert settings.verda is None
    assert settings.scaleway is None
    assert settings.file_storage is None


def test_transcription_settings_remote():
    """Test TranscriptionSettings accepts remote backend."""
    settings = TranscriptionSettings(
        backend="remote",
        endpoint_url="http://192.168.178.20:8000",
        worker_api_key="wk-secret",
    )
    assert settings.backend == "remote"
    assert settings.endpoint_url == "http://192.168.178.20:8000"
    assert settings.worker_api_key == "wk-secret"


def test_transcription_settings_koyeb():
    """Test TranscriptionSettings with Koyeb config."""
    settings = TranscriptionSettings(
        backend="koyeb",
        endpoint_url="https://hark.koyeb.app",
        worker_api_key="wk-secret",
        koyeb=KoyebProviderSettings(token="koyeb-tok"),
    )
    assert settings.backend == "koyeb"
    assert settings.koyeb is not None
    assert settings.koyeb.token == "koyeb-tok"
    assert settings.worker_api_key == "wk-secret"


def test_transcription_settings_verda_with_file_storage():
    """Test TranscriptionSettings with Verda and S3 storage."""
    settings = TranscriptionSettings(
        backend="verda",
        endpoint_url="https://containers.datacrunch.io/hark",
        verda=VerdaProviderSettings(api_key="dc_inf_test", poll_interval=10),
        file_storage=FileStorageSettings(
            endpoint="https://s3.example.com",
            bucket="hark-audio",
            access_key="AKID",
            secret_key="SECRET",
        ),
    )
    assert settings.verda is not None
    assert settings.verda.api_key == "dc_inf_test"
    assert settings.verda.poll_interval == 10
    assert settings.file_storage is not None
    assert settings.file_storage.bucket == "hark-audio"


def test_transcription_settings_verda_poll_interval_validation():
    """Test VerdaProviderSettings poll_interval range validation."""
    with pytest.raises(ValidationError):
        VerdaProviderSettings(api_key="x", poll_interval=0)
    with pytest.raises(ValidationError):
        VerdaProviderSettings(api_key="x", poll_interval=61)


def test_transcription_settings_remote_timeout_validation():
    """Test remote_timeout minimum validation."""
    with pytest.raises(ValidationError):
        TranscriptionSettings(backend="local", remote_timeout=30)


def test_harkd_settings_includes_transcription():
    """Test HarkdSettings has transcription field with defaults."""
    settings = HarkdSettings()
    assert isinstance(settings.transcription, TranscriptionSettings)
    assert settings.transcription.backend == "local"


def test_harkd_settings_transcription_from_yaml():
    """Test TranscriptionSettings loading from YAML file."""
    yaml_content = """
transcription:
  backend: koyeb
  endpoint_url: https://hark.koyeb.app
  worker_api_key: wk-from-yaml
  remote_timeout: 1800
  max_retries: 3
  fallback_to_local: false
  koyeb:
    token: koyeb-yaml-token
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = Path(f.name)

    try:
        settings = HarkdSettings.from_yaml(yaml_path)
        assert settings.transcription.backend == "koyeb"
        assert settings.transcription.endpoint_url == "https://hark.koyeb.app"
        assert settings.transcription.worker_api_key == "wk-from-yaml"
        assert settings.transcription.remote_timeout == 1800
        assert settings.transcription.max_retries == 3
        assert settings.transcription.fallback_to_local is False
        assert settings.transcription.koyeb is not None
        assert settings.transcription.koyeb.token == "koyeb-yaml-token"
    finally:
        yaml_path.unlink()


def test_harkd_settings_transcription_verda_from_yaml():
    """Test Verda TranscriptionSettings loading from YAML with file_storage."""
    yaml_content = """
transcription:
  backend: verda
  endpoint_url: https://containers.datacrunch.io/hark
  verda:
    api_key: dc_inf_yaml
    poll_interval: 10
  file_storage:
    endpoint: https://s3.example.com
    bucket: hark-audio
    access_key: AKID
    secret_key: SECRET
    region: eu-west-1
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = Path(f.name)

    try:
        settings = HarkdSettings.from_yaml(yaml_path)
        assert settings.transcription.backend == "verda"
        assert settings.transcription.verda is not None
        assert settings.transcription.verda.api_key == "dc_inf_yaml"
        assert settings.transcription.verda.poll_interval == 10
        assert settings.transcription.file_storage is not None
        assert settings.transcription.file_storage.bucket == "hark-audio"
        assert settings.transcription.file_storage.region == "eu-west-1"
    finally:
        yaml_path.unlink()


def test_harkd_settings_transcription_env_override(monkeypatch):
    """Test TranscriptionSettings can be overridden via env vars."""
    monkeypatch.setenv("HARKD_TRANSCRIPTION__BACKEND", "koyeb")
    monkeypatch.setenv("HARKD_TRANSCRIPTION__ENDPOINT_URL", "https://hark.koyeb.app")
    monkeypatch.setenv("HARKD_TRANSCRIPTION__WORKER_API_KEY", "env-secret")
    monkeypatch.setenv("HARKD_TRANSCRIPTION__FALLBACK_TO_LOCAL", "false")

    settings = HarkdSettings()
    assert settings.transcription.backend == "koyeb"
    assert settings.transcription.endpoint_url == "https://hark.koyeb.app"
    assert settings.transcription.worker_api_key == "env-secret"
    assert settings.transcription.fallback_to_local is False


def test_speaker_match_threshold_env_override(monkeypatch):
    """Test speaker_match_threshold can be set via environment variable."""
    monkeypatch.setenv("HARKD_RECORDING__SPEAKER_MATCH_THRESHOLD", "0.85")

    settings = HarkdSettings()
    assert settings.recording.speaker_match_threshold == 0.85


def test_speaker_match_threshold_from_yaml():
    """Test speaker_match_threshold loading from YAML file."""
    yaml_content = """
recording:
  speaker_match_threshold: 0.6
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = Path(f.name)

    try:
        settings = HarkdSettings.from_yaml(yaml_path)
        assert settings.recording.speaker_match_threshold == 0.6
    finally:
        yaml_path.unlink()


# --- Infrastructure settings tests ---


def test_infra_settings_defaults():
    """Test InfraSettings with defaults."""
    settings = InfraSettings()
    assert settings.docker_image == "ghcr.io/harkhq/harkd/worker:latest"
    assert settings.idle_timeout == 300
    assert settings.max_runtime == 0
    assert settings.provisioning_timeout == 600


def test_infra_settings_custom():
    """Test InfraSettings with custom values."""
    settings = InfraSettings(
        docker_image="custom:v1",
        idle_timeout=600,
        max_runtime=3600,
        provisioning_timeout=120,
    )
    assert settings.docker_image == "custom:v1"
    assert settings.idle_timeout == 600
    assert settings.max_runtime == 3600
    assert settings.provisioning_timeout == 120


def test_infra_settings_provisioning_timeout_validation():
    """Test provisioning_timeout minimum validation."""
    with pytest.raises(ValidationError):
        InfraSettings(provisioning_timeout=30)


def test_koyeb_infra_settings():
    """Test KoyebInfraSettings."""
    settings = KoyebInfraSettings(api_token="tok")
    assert settings.api_token == "tok"
    assert settings.region == "fra"
    assert settings.instance_type == "gpu-nvidia-rtx-4000-sff-ada"
    assert settings.app_name == "hark-worker"
    assert settings.use_native_scale_to_zero is True


def test_scaleway_infra_settings():
    """Test ScalewayInfraSettings."""
    settings = ScalewayInfraSettings(
        secret_key="key",
        organization_id="org",
        project_id="proj",
    )
    assert settings.zone == "fr-par-2"
    assert settings.instance_type == "L4-1-24G"
    assert settings.image_id is None


def test_datacrunch_infra_settings():
    """Test DataCrunchInfraSettings."""
    settings = DataCrunchInfraSettings(
        client_id="cid",
        client_secret="csecret",
    )
    assert settings.instance_type == "1L40S.6V"
    assert settings.location == "FIN-01"
    assert settings.ssh_key_ids == []
    assert settings.os_volume_id is None


def test_transcription_settings_managed_defaults():
    """Test TranscriptionSettings managed defaults."""
    settings = TranscriptionSettings()
    assert settings.managed is False
    assert settings.infra is None
    assert settings.koyeb_infra is None
    assert settings.scaleway_infra is None
    assert settings.datacrunch_infra is None


def test_transcription_settings_managed_koyeb_valid():
    """Test valid managed Koyeb config."""
    settings = TranscriptionSettings(
        backend="koyeb",
        managed=True,
        worker_api_key="key",
        infra=InfraSettings(),
        koyeb_infra=KoyebInfraSettings(api_token="tok"),
    )
    assert settings.managed is True


def test_transcription_settings_managed_missing_infra():
    """Test managed=True without infra settings raises."""
    with pytest.raises(ValidationError, match="infra"):
        TranscriptionSettings(
            backend="koyeb",
            managed=True,
        )


def test_transcription_settings_managed_koyeb_missing_koyeb_infra():
    """Test managed koyeb without koyeb_infra raises."""
    with pytest.raises(ValidationError, match="koyeb_infra"):
        TranscriptionSettings(
            backend="koyeb",
            managed=True,
            infra=InfraSettings(),
        )


def test_transcription_settings_managed_scaleway_missing_scaleway_infra():
    """Test managed scaleway without scaleway_infra raises."""
    with pytest.raises(ValidationError, match="scaleway_infra"):
        TranscriptionSettings(
            backend="scaleway",
            managed=True,
            infra=InfraSettings(),
        )


def test_transcription_settings_managed_verda_missing_datacrunch_infra():
    """Test managed verda without datacrunch_infra raises."""
    with pytest.raises(ValidationError, match="datacrunch_infra"):
        TranscriptionSettings(
            backend="verda",
            managed=True,
            infra=InfraSettings(),
        )


def test_transcription_settings_managed_local_no_infra_needed():
    """Test managed=True with local backend needs no infra config."""
    settings = TranscriptionSettings(
        backend="local",
        managed=True,
    )
    assert settings.managed is True
    assert settings.infra is None


def test_transcription_settings_managed_remote_no_infra_needed():
    """Test managed=True with remote backend needs no infra config."""
    settings = TranscriptionSettings(
        backend="remote",
        managed=True,
        endpoint_url="http://192.168.178.20:8000",
    )
    assert settings.managed is True
    assert settings.infra is None


def test_transcription_settings_remote_from_yaml():
    """Test remote TranscriptionSettings loading from YAML file."""
    yaml_content = """
transcription:
  backend: remote
  endpoint_url: http://192.168.178.20:8000
  worker_api_key: wk-from-yaml
  remote_timeout: 1800
  fallback_to_local: true
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = Path(f.name)

    try:
        settings = HarkdSettings.from_yaml(yaml_path)
        assert settings.transcription.backend == "remote"
        assert settings.transcription.endpoint_url == "http://192.168.178.20:8000"
        assert settings.transcription.worker_api_key == "wk-from-yaml"
        assert settings.transcription.remote_timeout == 1800
        assert settings.transcription.fallback_to_local is True
        # No provider-specific config needed
        assert settings.transcription.koyeb is None
        assert settings.transcription.scaleway is None
    finally:
        yaml_path.unlink()


def test_transcription_settings_managed_from_yaml():
    """Test managed config loading from YAML."""
    yaml_content = """
transcription:
  backend: koyeb
  managed: true
  worker_api_key: wk-yaml
  infra:
    docker_image: ghcr.io/test/worker:latest
    idle_timeout: 600
  koyeb_infra:
    api_token: koyeb-yaml-token
    region: was
    use_native_scale_to_zero: false
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = Path(f.name)

    try:
        settings = HarkdSettings.from_yaml(yaml_path)
        assert settings.transcription.managed is True
        assert settings.transcription.infra is not None
        assert settings.transcription.infra.docker_image == "ghcr.io/test/worker:latest"
        assert settings.transcription.infra.idle_timeout == 600
        assert settings.transcription.koyeb_infra is not None
        assert settings.transcription.koyeb_infra.api_token == "koyeb-yaml-token"
        assert settings.transcription.koyeb_infra.region == "was"
        assert settings.transcription.koyeb_infra.use_native_scale_to_zero is False
    finally:
        yaml_path.unlink()
