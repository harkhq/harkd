"""Tests for settings API models."""

from harkd.api.models.settings import Settings


class TestSettings:
    """Test Settings model."""

    def test_settings_all_fields(self):
        """Test creating settings with all fields."""
        settings = Settings(
            model="large-v3",
            word_timestamps=False,
            language="auto",
            diarization=True,
            noise_reduction=True,
            normalization=True,
            mic_gain=2.0,
        )
        assert settings.model == "large-v3"
        assert settings.word_timestamps is False
        assert settings.language == "auto"
        assert settings.diarization is True
        assert settings.noise_reduction is True
        assert settings.normalization is True
        assert settings.mic_gain == 2.0

    def test_settings_custom_values(self):
        """Test creating settings with custom values."""
        settings = Settings(
            model="small",
            word_timestamps=True,
            language="en",
            diarization=False,
            noise_reduction=False,
            normalization=False,
            mic_gain=3.5,
        )
        assert settings.model == "small"
        assert settings.word_timestamps is True
        assert settings.language == "en"
        assert settings.diarization is False
        assert settings.mic_gain == 3.5

    def test_settings_no_device_or_ram_fields(self):
        """Test that old computed fields (device, ram_usage) are not present."""
        settings = Settings(
            model="base",
            word_timestamps=False,
            language="auto",
            diarization=True,
            noise_reduction=True,
            normalization=True,
            mic_gain=2.0,
        )
        assert not hasattr(settings, "device")
        assert not hasattr(settings, "ram_usage")

    def test_settings_no_input_source_field(self):
        """Test that input_source field has been removed."""
        assert "input_source" not in Settings.model_fields
