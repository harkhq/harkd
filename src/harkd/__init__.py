"""harkd - FastAPI daemon for hark voice recording and transcription."""

import os

# Force CPU-only to avoid CUDA initialization issues in server context
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# NOTE: torch patch moved to transcription_worker.py to avoid "poison fork" issue
# PyTorch must NOT be initialized in the parent process before subprocess spawning

__version__ = "0.1.0"

__all__ = ["__version__"]
