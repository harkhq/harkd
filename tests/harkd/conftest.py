"""Shared pytest fixtures for harkd tests."""

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_whisper_model(monkeypatch):
    """Mock whisperx.load_model for transcriber tests."""
    # Create a mock torch module (to prevent real torch import)
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    # Create a mock whisperx module
    mock_whisperx = MagicMock()

    # Mock the load_model function
    mock_load_model = MagicMock()
    mock_whisperx.load_model = mock_load_model

    # Mock load_audio function
    mock_whisperx.load_audio = MagicMock(return_value=MagicMock())

    # Mock load_align_model function
    mock_whisperx.load_align_model = MagicMock()

    # Mock align function
    mock_whisperx.align = MagicMock()

    # Inject into sys.modules
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "whisperx", mock_whisperx)

    return mock_load_model
