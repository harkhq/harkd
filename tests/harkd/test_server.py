"""Tests for server startup validation."""

import pytest

from harkd.config import HarkdSettings, LLMSettings, RecordingDefaults
from harkd.server import _validate_settings


class TestValidateSettings:
    """Tests for _validate_settings startup checks."""

    def test_diarization_enabled_no_token_raises(self):
        """Test that diarization without hf_token raises SystemExit."""
        settings = HarkdSettings(
            recording=RecordingDefaults(diarization=True),
            hf_token=None,
        )

        with pytest.raises(SystemExit) as exc_info:
            _validate_settings(settings)

        assert "Diarization is enabled" in str(exc_info.value)
        assert "HuggingFace token" in str(exc_info.value)

    def test_diarization_enabled_with_token_passes(self):
        """Test that diarization with hf_token passes validation."""
        settings = HarkdSettings(
            recording=RecordingDefaults(diarization=True),
            hf_token="hf_test123",
        )

        # Should not raise
        _validate_settings(settings)

    def test_diarization_disabled_no_token_passes(self):
        """Test that disabled diarization without token passes."""
        settings = HarkdSettings(
            recording=RecordingDefaults(diarization=False),
            hf_token=None,
        )

        # Should not raise
        _validate_settings(settings)

    def test_llm_enabled_cloud_provider_no_key_raises(self):
        """Test that LLM with cloud provider and no API key raises SystemExit."""
        settings = HarkdSettings(
            recording=RecordingDefaults(diarization=False),
            llm=LLMSettings(enabled=True, provider="openai", api_key=None),
        )

        with pytest.raises(SystemExit) as exc_info:
            _validate_settings(settings)

        assert "LLM is enabled" in str(exc_info.value)
        assert "openai" in str(exc_info.value)

    def test_llm_enabled_anthropic_no_key_raises(self):
        """Test that LLM with Anthropic and no API key raises SystemExit."""
        settings = HarkdSettings(
            recording=RecordingDefaults(diarization=False),
            llm=LLMSettings(enabled=True, provider="anthropic", api_key=None),
        )

        with pytest.raises(SystemExit) as exc_info:
            _validate_settings(settings)

        assert "anthropic" in str(exc_info.value)

    def test_llm_enabled_ollama_no_key_passes(self):
        """Test that LLM with Ollama and no API key passes."""
        settings = HarkdSettings(
            recording=RecordingDefaults(diarization=False),
            llm=LLMSettings(enabled=True, provider="ollama", api_key=None),
        )

        # Should not raise — Ollama doesn't need an API key
        _validate_settings(settings)

    def test_llm_enabled_with_key_passes(self):
        """Test that LLM with cloud provider and API key passes."""
        settings = HarkdSettings(
            recording=RecordingDefaults(diarization=False),
            llm=LLMSettings(enabled=True, provider="openai", api_key="sk-test"),
        )

        # Should not raise
        _validate_settings(settings)

    def test_llm_disabled_no_key_passes(self):
        """Test that disabled LLM without API key passes."""
        settings = HarkdSettings(
            recording=RecordingDefaults(diarization=False),
            llm=LLMSettings(enabled=False, provider="openai", api_key=None),
        )

        # Should not raise
        _validate_settings(settings)

    def test_both_diarization_and_llm_validated(self):
        """Test that both validations run."""
        # Diarization valid, LLM invalid
        settings = HarkdSettings(
            recording=RecordingDefaults(diarization=False),
            hf_token=None,
            llm=LLMSettings(enabled=True, provider="google", api_key=None),
        )

        with pytest.raises(SystemExit) as exc_info:
            _validate_settings(settings)

        assert "google" in str(exc_info.value)
