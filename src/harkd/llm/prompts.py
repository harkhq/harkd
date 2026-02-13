"""Prompt management with default templates and user-customizable overrides."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# fmt: off
DEFAULT_PROMPTS: dict[str, str] = {
    "meeting_minutes": (
        "You are an expert meeting assistant. Analyze the following transcript"
        " and extract structured meeting minutes.\n\n"
        "Transcript:\n{transcript}\n\n"
        "Speakers: {speakers}\n"
        "Language: {language}\n\n"
        "Respond with ONLY a JSON object with these fields:\n"
        '- "executive_summary": list of 1-3 bullet-point strings summarizing'
        " the key takeaways\n"
        '- "meeting_notes": list of objects with "topic" and "content" keys\n'
        '- "tasks": list of objects with "task", "assignee" (speaker name or'
        ' null), and "due" (date or null) keys\n'
        '- "decisions": list of strings describing decisions made\n\n'
        "If the transcript is a monologue or doesn't have clear meeting"
        " structure, still extract whatever structure you can (summary, notes,"
        " tasks if mentioned)."
    ),
}
# fmt: on


class PromptManager:
    """Manages prompt templates with optional user overrides.

    Looks for custom prompts in a user-specified directory first,
    then falls back to built-in defaults.
    """

    def __init__(self, custom_dir: Path | None = None):
        """Initialize prompt manager.

        Args:
            custom_dir: Optional directory containing custom prompt .txt files
        """
        self._custom_dir = custom_dir

    def get(self, name: str) -> str:
        """Get a prompt template by name.

        Args:
            name: Prompt name (e.g. "meeting_minutes")

        Returns:
            Prompt template string with {placeholders}

        Raises:
            KeyError: If no prompt found with that name
        """
        # Check custom directory first
        if self._custom_dir:
            custom_path = self._custom_dir / f"{name}.txt"
            if custom_path.is_file():
                logger.debug(f"Using custom prompt: {custom_path}")
                return custom_path.read_text(encoding="utf-8")

        # Fall back to defaults
        if name in DEFAULT_PROMPTS:
            return DEFAULT_PROMPTS[name]

        raise KeyError(f"No prompt found: {name}")
