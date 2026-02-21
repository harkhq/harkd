"""Auto-detect system capabilities for transcription and diarization."""

import logging
import os

logger = logging.getLogger(__name__)


def detect_cpu_threads() -> int:
    """Return the number of CPU threads to use for model inference.

    Reserves 2 cores for the OS and daemon, minimum 1.
    """
    threads = max(1, (os.cpu_count() or 4) - 2)
    logger.debug("Auto-detected CPU threads: %d", threads)
    return threads


def detect_batch_size() -> int:
    """Return a batch size appropriate for the available system RAM.

    Heuristic: 1 batch slot per ~2 GB of RAM, clamped to [1, 32].
    Falls back to 16 on any error.
    """
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        total_gb = (page_size * page_count) / (1024**3)
        batch_size = min(32, max(1, int(total_gb // 2)))
    except (ValueError, OSError):
        batch_size = 16
    logger.debug("Auto-detected batch size: %d", batch_size)
    return batch_size
