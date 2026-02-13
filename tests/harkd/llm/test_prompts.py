"""Tests for prompt management."""

import pytest

from harkd.llm.prompts import DEFAULT_PROMPTS, PromptManager


class TestPromptManager:
    """Tests for PromptManager."""

    def test_default_prompts_returned(self):
        """Test that default prompts are returned when no custom dir."""
        pm = PromptManager()
        prompt = pm.get("meeting_minutes")

        assert "{transcript}" in prompt
        assert "{speakers}" in prompt
        assert "{language}" in prompt

    def test_missing_prompt_raises_key_error(self):
        """Test that unknown prompt name raises KeyError."""
        pm = PromptManager()

        with pytest.raises(KeyError, match="No prompt found: nonexistent"):
            pm.get("nonexistent")

    def test_custom_prompt_overrides_default(self, tmp_path):
        """Test that custom prompt file overrides default."""
        # Create custom prompt
        custom_prompt = "Custom meeting minutes: {transcript}"
        prompt_file = tmp_path / "meeting_minutes.txt"
        prompt_file.write_text(custom_prompt)

        pm = PromptManager(custom_dir=tmp_path)
        result = pm.get("meeting_minutes")

        assert result == custom_prompt
        assert result != DEFAULT_PROMPTS["meeting_minutes"]

    def test_missing_custom_falls_back_to_default(self, tmp_path):
        """Test that missing custom prompt falls back to default."""
        # Custom dir exists but doesn't have the prompt file
        pm = PromptManager(custom_dir=tmp_path)
        result = pm.get("meeting_minutes")

        assert result == DEFAULT_PROMPTS["meeting_minutes"]

    def test_prompt_placeholder_substitution(self):
        """Test that prompt placeholders can be filled in."""
        pm = PromptManager()
        template = pm.get("meeting_minutes")

        filled = template.format(
            transcript="Hello world",
            speakers="Alice, Bob",
            language="en",
        )

        assert "Hello world" in filled
        assert "Alice, Bob" in filled
        assert "en" in filled
        assert "{transcript}" not in filled

    def test_default_prompts_has_meeting_minutes(self):
        """Test that DEFAULT_PROMPTS contains meeting_minutes."""
        assert "meeting_minutes" in DEFAULT_PROMPTS
        assert len(DEFAULT_PROMPTS["meeting_minutes"]) > 100  # Non-trivial
