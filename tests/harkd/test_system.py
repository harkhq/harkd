"""Tests for audio._system helpers."""

from __future__ import annotations

from unittest.mock import patch

from harkd.audio._system import detect_batch_size, detect_cpu_threads


class TestDetectCpuThreads:
    """Tests for detect_cpu_threads."""

    def test_reserves_two_cores(self):
        """Should return cpu_count - 2."""
        with patch("harkd.audio._system.os.cpu_count", return_value=20):
            assert detect_cpu_threads() == 18

    def test_minimum_one(self):
        """Should never return less than 1."""
        with patch("harkd.audio._system.os.cpu_count", return_value=1):
            assert detect_cpu_threads() == 1

        with patch("harkd.audio._system.os.cpu_count", return_value=2):
            assert detect_cpu_threads() == 1

    def test_none_cpu_count_uses_fallback(self):
        """Should fall back to 4 cores when os.cpu_count() returns None."""
        with patch("harkd.audio._system.os.cpu_count", return_value=None):
            assert detect_cpu_threads() == 2  # (4 - 2)

    def test_typical_values(self):
        """Spot-check a few common core counts."""
        for cores, expected in [(4, 2), (8, 6), (16, 14), (32, 30)]:
            with patch("harkd.audio._system.os.cpu_count", return_value=cores):
                assert detect_cpu_threads() == expected, f"Failed for {cores} cores"


class TestDetectBatchSize:
    """Tests for detect_batch_size."""

    def test_scales_with_ram(self):
        """Batch size should be ~total_gb // 2."""
        page_size = 4096
        # 32 GB -> 32 // 2 = 16
        pages_32gb = (32 * 1024**3) // page_size
        with patch("harkd.audio._system.os.sysconf", side_effect=[page_size, pages_32gb]):
            assert detect_batch_size() == 16

    def test_caps_at_32(self):
        """Batch size should never exceed 32."""
        page_size = 4096
        pages_128gb = (128 * 1024**3) // page_size
        with patch("harkd.audio._system.os.sysconf", side_effect=[page_size, pages_128gb]):
            assert detect_batch_size() == 32

    def test_minimum_one(self):
        """Batch size should never go below 1."""
        page_size = 4096
        pages_1gb = (1 * 1024**3) // page_size
        with patch("harkd.audio._system.os.sysconf", side_effect=[page_size, pages_1gb]):
            assert detect_batch_size() == 1

    def test_fallback_on_error(self):
        """Should return 16 if sysconf raises."""
        with patch("harkd.audio._system.os.sysconf", side_effect=ValueError("unsupported")):
            assert detect_batch_size() == 16

    def test_fallback_on_os_error(self):
        """Should return 16 if sysconf raises OSError (e.g. macOS)."""
        with patch("harkd.audio._system.os.sysconf", side_effect=OSError("not available")):
            assert detect_batch_size() == 16
