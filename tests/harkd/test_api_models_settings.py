"""Tests for settings API models."""

import pytest
from pydantic import ValidationError

from harkd.api.models.settings import Settings


class TestSettings:
    """Test Settings model."""

    def test_settings_all_fields(self):
        """Test creating settings with all fields."""
        settings = Settings(
            model="large-v3",
            word_timestamps=False,
            language="auto",
            input_source="both",
            diarization=True,
            noise_reduction=True,
            normalization=True,
        )
        assert settings.model == "large-v3"
        assert settings.word_timestamps is False
        assert settings.language == "auto"
        assert settings.input_source == "both"
        assert settings.diarization is True
        assert settings.noise_reduction is True
        assert settings.normalization is True

    def test_settings_custom_values(self):
        """Test creating settings with custom values."""
        settings = Settings(
            model="small",
            word_timestamps=True,
            language="en",
            input_source="mic",
            diarization=False,
            noise_reduction=False,
            normalization=False,
        )
        assert settings.model == "small"
        assert settings.word_timestamps is True
        assert settings.language == "en"
        assert settings.input_source == "mic"
        assert settings.diarization is False

    def test_settings_invalid_input_source(self):
        """Test validation fails for invalid input source."""
        with pytest.raises(ValidationError):
            Settings(
                model="base",
                word_timestamps=False,
                language="auto",
                input_source="invalid",
                diarization=True,
                noise_reduction=True,
                normalization=True,
            )

    def test_settings_valid_input_sources(self):
        """Test all valid input sources are accepted."""
        for source in ["mic", "speaker", "both"]:
            settings = Settings(
                model="base",
                word_timestamps=False,
                language="auto",
                input_source=source,
                diarization=True,
                noise_reduction=True,
                normalization=True,
            )
            assert settings.input_source == source

    def test_settings_no_device_or_ram_fields(self):
        """Test that old computed fields (device, ram_usage) are not present."""
        settings = Settings(
            model="base",
            word_timestamps=False,
            language="auto",
            input_source="both",
            diarization=True,
            noise_reduction=True,
            normalization=True,
        )
        assert not hasattr(settings, "device")
        assert not hasattr(settings, "ram_usage")
