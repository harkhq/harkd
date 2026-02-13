"""Platform detection utilities for harkd."""

import platform

__all__ = ["is_linux", "is_macos", "is_windows"]


def is_linux() -> bool:
    """Check if running on Linux."""
    return platform.system() == "Linux"


def is_macos() -> bool:
    """Check if running on macOS."""
    return platform.system() == "Darwin"


def is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system() == "Windows"
