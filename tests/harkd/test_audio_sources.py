"""Tests for audio source detection."""

from unittest.mock import MagicMock, patch

import pytest

from harkd.audio.sources import (
    AudioSourceInfo,
    _get_default_sink_monitor,
    _is_loopback_device,
    find_loopback_device,
)


class TestIsLoopbackDevice:
    """Tests for _is_loopback_device detection."""

    @pytest.mark.parametrize(
        "device_name",
        [
            "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor",
            "Monitor of Built-in Audio Analog Stereo",
            "BlackHole 2ch",
            "blackhole 16ch",
            "Stereo Mix",
            "stereo mix (Realtek HD Audio)",
            "Soundflower (2ch)",
            "What U Hear (Realtek HD Audio)",
            "My Loopback Device",
        ],
    )
    def test_loopback_devices_detected(self, device_name):
        """Test that known loopback device names are detected."""
        assert _is_loopback_device(device_name) is True

    @pytest.mark.parametrize(
        "device_name",
        [
            "Built-in Microphone",
            "USB Microphone",
            "Blue Yeti",
            "Default Input",
            "Realtek HD Audio Input",
            "External Headset Mic",
        ],
    )
    def test_regular_devices_not_detected(self, device_name):
        """Test that regular microphone names are not detected as loopback."""
        assert _is_loopback_device(device_name) is False


# ============================================================================
# Regression tests for bugs found during real-world testing
# ============================================================================


class TestGetDefaultSinkMonitor:
    """Regression: wrong loopback device selected.

    Without default sink preference, the first .monitor device was selected
    (e.g. a webcam) instead of the actual audio output (e.g. USB speakerphone).
    """

    @patch("harkd.audio.sources.subprocess.run")
    def test_returns_monitor_name_for_default_sink(self, mock_run):
        """Must append .monitor to the default sink name."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="alsa_output.usb-ANKER_PowerConf.iec958-stereo\n",
        )
        result = _get_default_sink_monitor()
        assert result == "alsa_output.usb-ANKER_PowerConf.iec958-stereo.monitor"

    @patch("harkd.audio.sources.subprocess.run")
    def test_returns_none_when_pactl_fails(self, mock_run):
        """Must return None when pactl is not available."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert _get_default_sink_monitor() is None

    @patch("harkd.audio.sources.subprocess.run")
    def test_returns_none_when_pactl_not_found(self, mock_run):
        """Must return None when pactl binary is missing."""
        mock_run.side_effect = FileNotFoundError
        assert _get_default_sink_monitor() is None


class TestFindLoopbackDevicePreference:
    """Regression: default sink monitor must be preferred over arbitrary monitors.

    On a system with multiple monitors (webcam, HDMI, USB speakerphone),
    the monitor matching the default PulseAudio sink should be selected.
    """

    @patch("harkd.audio.sources._get_default_sink_monitor")
    @patch("harkd.audio.sources.list_loopback_devices")
    @patch("harkd.audio.sources.is_linux", return_value=True)
    @patch("harkd.audio.sources.is_macos", return_value=False)
    @patch("harkd.audio.sources.is_windows", return_value=False)
    def test_linux_prefers_default_sink_monitor(
        self, _win, _mac, _linux, mock_list, mock_default_sink
    ):
        """On Linux, must select the monitor matching the default sink."""
        webcam = AudioSourceInfo(
            device_index=17,
            name="alsa_output.webcam.monitor",
            channels=2,
            sample_rate=44100,
            is_loopback=True,
            pulse_source="alsa_output.webcam.monitor",
        )
        anker = AudioSourceInfo(
            device_index=17,
            name="alsa_output.anker.monitor",
            channels=2,
            sample_rate=44100,
            is_loopback=True,
            pulse_source="alsa_output.anker.monitor",
        )
        mock_list.return_value = [webcam, anker]
        mock_default_sink.return_value = "alsa_output.anker.monitor"

        result = find_loopback_device()
        assert result is anker

    @patch("harkd.audio.sources._get_default_sink_monitor")
    @patch("harkd.audio.sources.list_loopback_devices")
    @patch("harkd.audio.sources.is_linux", return_value=True)
    @patch("harkd.audio.sources.is_macos", return_value=False)
    @patch("harkd.audio.sources.is_windows", return_value=False)
    def test_linux_falls_back_to_first_monitor(
        self, _win, _mac, _linux, mock_list, mock_default_sink
    ):
        """On Linux, must fall back to first .monitor if default sink not found."""
        webcam = AudioSourceInfo(
            device_index=17,
            name="alsa_output.webcam.monitor",
            channels=2,
            sample_rate=44100,
            is_loopback=True,
            pulse_source="alsa_output.webcam.monitor",
        )
        mock_list.return_value = [webcam]
        mock_default_sink.return_value = "alsa_output.nonexistent.monitor"

        result = find_loopback_device()
        assert result is webcam


class TestPulseAudioMonitorDevices:
    """Regression: ALSA/PulseAudio device mismatch caused segfaults.

    PulseAudio monitors must use the 'pulse' sounddevice with PULSE_SOURCE,
    NOT map to ALSA hardware device indices by name. The pulse_source field
    must be set for PulseAudio-discovered monitors.
    """

    @patch("harkd.audio.sources._get_default_sink_monitor")
    @patch("harkd.audio.sources.list_loopback_devices")
    @patch("harkd.audio.sources.is_linux", return_value=True)
    @patch("harkd.audio.sources.is_macos", return_value=False)
    @patch("harkd.audio.sources.is_windows", return_value=False)
    def test_pulse_monitor_has_pulse_source_field(
        self, _win, _mac, _linux, mock_list, mock_default_sink
    ):
        """PulseAudio monitors must have pulse_source set (for PULSE_SOURCE env var)."""
        monitor = AudioSourceInfo(
            device_index=17,
            name="alsa_output.device.monitor",
            channels=2,
            sample_rate=44100,
            is_loopback=True,
            pulse_source="alsa_output.device.monitor",
        )
        mock_list.return_value = [monitor]
        mock_default_sink.return_value = "alsa_output.device.monitor"

        result = find_loopback_device()
        assert result is not None
        assert result.pulse_source == "alsa_output.device.monitor"


class TestMacOSWindowsNosPulseSource:
    """Verify macOS/Windows loopback devices don't use PULSE_SOURCE.

    Only Linux PulseAudio monitors need PULSE_SOURCE routing. macOS (BlackHole)
    and Windows (Stereo Mix) use real device indices directly.
    """

    @patch("harkd.audio.sources.list_loopback_devices")
    @patch("harkd.audio.sources.is_linux", return_value=False)
    @patch("harkd.audio.sources.is_macos", return_value=True)
    @patch("harkd.audio.sources.is_windows", return_value=False)
    def test_macos_blackhole_no_pulse_source(self, _win, _mac, _linux, mock_list):
        """macOS BlackHole devices must NOT have pulse_source set."""
        blackhole = AudioSourceInfo(
            device_index=5,
            name="BlackHole 2ch",
            channels=2,
            sample_rate=48000,
            is_loopback=True,
            pulse_source=None,
        )
        mock_list.return_value = [blackhole]

        result = find_loopback_device()
        assert result is blackhole
        assert result.pulse_source is None

    @patch("harkd.audio.sources.list_loopback_devices")
    @patch("harkd.audio.sources.is_linux", return_value=False)
    @patch("harkd.audio.sources.is_macos", return_value=False)
    @patch("harkd.audio.sources.is_windows", return_value=True)
    def test_windows_stereo_mix_no_pulse_source(self, _win, _mac, _linux, mock_list):
        """Windows Stereo Mix devices must NOT have pulse_source set."""
        stereo_mix = AudioSourceInfo(
            device_index=3,
            name="Stereo Mix (Realtek HD Audio)",
            channels=2,
            sample_rate=44100,
            is_loopback=True,
            pulse_source=None,
        )
        mock_list.return_value = [stereo_mix]

        result = find_loopback_device()
        assert result is stereo_mix
        assert result.pulse_source is None
