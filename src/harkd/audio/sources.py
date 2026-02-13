"""Audio source detection for microphones and loopback devices."""

import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Any, cast

import sounddevice as sd

from harkd.audio.platform import is_linux, is_macos, is_windows

__all__ = [
    "AudioSourceInfo",
    "find_microphone",
    "find_loopback_device",
    "list_loopback_devices",
    "get_loopback_instructions",
]

logger = logging.getLogger(__name__)


@dataclass
class AudioSourceInfo:
    """Information about an audio source."""

    device_index: int | None  # sounddevice index
    name: str
    channels: int
    sample_rate: float
    is_loopback: bool
    pulse_source: str | None = None  # PulseAudio source name (set PULSE_SOURCE before recording)


def _is_loopback_device(device_name: str) -> bool:
    """Check if device name indicates a loopback/monitor source.

    Args:
        device_name: Name of the audio device

    Returns:
        True if device appears to be a loopback source
    """
    name_lower = device_name.lower()

    # PulseAudio/PipeWire patterns (Linux)
    if ".monitor" in name_lower:
        return True
    if "monitor of" in name_lower:
        return True

    # Common patterns across platforms
    if re.search(r"\bloopback\b", name_lower):
        return True
    if "stereo mix" in name_lower:  # Windows
        return True
    if "what u hear" in name_lower:  # Windows
        return True

    # macOS patterns (BlackHole, etc.)
    if "blackhole" in name_lower:
        return True
    return "soundflower" in name_lower


def find_microphone() -> AudioSourceInfo | None:
    """Find the default microphone device.

    Returns:
        AudioSourceInfo for default microphone, or None if not found
    """
    try:
        device_id = sd.default.device[0]  # Input device
        if device_id is None:
            logger.debug("No default input device configured")
            return None

        device = cast("dict[str, Any]", sd.query_devices(device_id))
        if device["max_input_channels"] > 0:
            logger.debug(
                f"Found microphone: {device['name']} "
                f"(device {device_id}, {device['max_input_channels']} channels)"
            )
            return AudioSourceInfo(
                device_index=int(device_id),
                name=str(device["name"]),
                channels=int(device["max_input_channels"]),
                sample_rate=float(device["default_samplerate"]),
                is_loopback=False,
            )
    except (sd.PortAudioError, KeyError, IndexError, TypeError) as e:
        logger.debug(f"Could not find microphone: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error finding microphone: {e}", exc_info=True)
        return None

    return None


def _find_pulse_device_index() -> int | None:
    """Find the sounddevice index for the 'pulse' device.

    The 'pulse' device routes audio through PulseAudio/PipeWire, which allows
    us to capture from monitor sources via PULSE_SOURCE env var.

    Returns:
        Device index for 'pulse', or None if not found
    """
    try:
        devices = cast("list[dict[str, Any]]", sd.query_devices())
        for idx, device in enumerate(devices):
            if device["max_input_channels"] > 0 and device["name"] == "pulse":
                return idx
    except Exception as e:
        logger.debug(f"Error finding pulse device: {e}")
    return None


def _list_pulseaudio_monitors() -> list[AudioSourceInfo]:
    """Query PulseAudio/PipeWire directly for monitor devices (Linux only).

    This is a fallback for when PortAudio doesn't see PulseAudio monitors
    (which happens when PortAudio uses ALSA backend instead of PulseAudio backend).

    Uses the 'pulse' sounddevice device with PULSE_SOURCE env var to route
    the correct monitor source. This avoids opening ALSA hardware devices
    directly, which can cause segfaults.

    Returns:
        List of monitor devices from PulseAudio/PipeWire
    """
    monitors = []

    # We need the 'pulse' device to route through PulseAudio
    pulse_idx = _find_pulse_device_index()
    if pulse_idx is None:
        logger.debug("No 'pulse' sounddevice found, cannot use PulseAudio monitors")
        return monitors

    try:
        pulse_info = cast("dict[str, Any]", sd.query_devices(pulse_idx))
        pulse_channels = int(pulse_info["max_input_channels"])
        pulse_sample_rate = float(pulse_info["default_samplerate"])
    except Exception:
        pulse_channels = 2
        pulse_sample_rate = 44100.0

    try:
        # Query PulseAudio/PipeWire sources
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )

        if result.returncode != 0:
            logger.debug("pactl not available or failed")
            return monitors

        # Parse output: each line is "INDEX NAME DRIVER FORMAT STATUS"
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue

            parts = line.split(maxsplit=4)
            if len(parts) < 2:
                continue

            source_name = parts[1]

            # Check if it's a monitor device
            if ".monitor" in source_name.lower():
                logger.debug(f"Found PulseAudio monitor: {source_name}")

                monitors.append(
                    AudioSourceInfo(
                        device_index=pulse_idx,
                        name=source_name,
                        channels=pulse_channels,
                        sample_rate=pulse_sample_rate,
                        is_loopback=True,
                        pulse_source=source_name,
                    )
                )

    except subprocess.TimeoutExpired:
        logger.debug("pactl command timed out")
    except FileNotFoundError:
        logger.debug("pactl command not found")
    except Exception as e:
        logger.debug(f"Error querying PulseAudio monitors: {e}")

    return monitors


def list_loopback_devices() -> list[AudioSourceInfo]:
    """List all available loopback devices.

    Returns:
        List of AudioSourceInfo for loopback devices
    """
    loopback_devices = []

    try:
        devices = cast("list[dict[str, Any]]", sd.query_devices())
        for idx, device in enumerate(devices):
            # Check if it's an input device and appears to be loopback
            if device["max_input_channels"] > 0 and _is_loopback_device(device["name"]):
                logger.debug(f"Found loopback device: {device['name']} (device {idx})")
                loopback_devices.append(
                    AudioSourceInfo(
                        device_index=idx,
                        name=str(device["name"]),
                        channels=int(device["max_input_channels"]),
                        sample_rate=float(device["default_samplerate"]),
                        is_loopback=True,
                    )
                )
    except (sd.PortAudioError, KeyError, TypeError) as e:
        logger.debug(f"Could not list loopback devices: {e}")
    except Exception as e:
        logger.error(f"Unexpected error listing loopback devices: {e}", exc_info=True)

    # Linux fallback: Query PulseAudio/PipeWire directly if no loopback devices found
    # This handles the case where PortAudio uses ALSA backend and doesn't see PA monitors
    if not loopback_devices and is_linux():
        logger.debug(
            "No loopback devices found via sounddevice, trying PulseAudio/PipeWire direct query"
        )
        pulseaudio_monitors = _list_pulseaudio_monitors()
        loopback_devices.extend(pulseaudio_monitors)

    logger.debug(f"Found {len(loopback_devices)} loopback device(s)")
    return loopback_devices


def _get_default_sink_monitor() -> str | None:
    """Get the monitor name for the default PulseAudio/PipeWire sink.

    Returns:
        Monitor source name (e.g. "alsa_output.xxx.monitor") or None
    """
    try:
        result = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            monitor = result.stdout.strip() + ".monitor"
            logger.debug(f"Default sink monitor: {monitor}")
            return monitor
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    except Exception as e:
        logger.debug(f"Error getting default sink: {e}")
    return None


def find_loopback_device() -> AudioSourceInfo | None:
    """Find a suitable loopback device for the current platform.

    Returns:
        AudioSourceInfo for a loopback device, or None if not found
    """
    # Get all loopback devices
    loopback_devices = list_loopback_devices()

    if not loopback_devices:
        return None

    # Platform-specific preference
    if is_linux():
        # Prefer the monitor of the default PulseAudio/PipeWire sink
        default_monitor = _get_default_sink_monitor()
        if default_monitor:
            for device in loopback_devices:
                if device.name == default_monitor:
                    logger.debug(f"Using default sink monitor: {device.name}")
                    return device

        # Fall back to first .monitor device
        for device in loopback_devices:
            if ".monitor" in device.name.lower():
                return device

    elif is_macos():
        # Prefer BlackHole on macOS
        for device in loopback_devices:
            if "blackhole" in device.name.lower():
                return device

    elif is_windows():
        # Prefer "Stereo Mix" on Windows
        for device in loopback_devices:
            if "stereo mix" in device.name.lower():
                return device

    # Return first available if no platform-specific match
    return loopback_devices[0] if loopback_devices else None


def get_loopback_instructions() -> str:
    """Get platform-specific instructions for enabling loopback.

    Returns:
        Instructions for enabling system audio capture
    """
    if is_linux():
        return (
            "Linux: Install PulseAudio or PipeWire. "
            "Monitor devices should be available automatically. "
            "Use 'pactl list sources' to see available sources."
        )
    elif is_macos():
        return (
            "macOS: Install a virtual audio device like BlackHole "
            "(brew install blackhole-2ch) or Soundflower."
        )
    elif is_windows():
        return (
            "Windows: Enable 'Stereo Mix' in Sound Settings > Recording tab. "
            "Right-click and select 'Show Disabled Devices' if not visible."
        )
    else:
        return "Platform not supported for loopback audio capture."
