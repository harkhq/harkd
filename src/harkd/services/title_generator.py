"""Title generation from transcripts."""

import re

__all__ = ["generate_title"]


def generate_title(transcript: str, max_length: int = 50) -> str:
    """Generate a title from transcript.

    Strategy:
    1. Take first sentence (up to first . ! ?)
    2. Truncate to max_length characters
    3. Clean up whitespace and newlines

    Args:
        transcript: Full transcript text
        max_length: Maximum title length (default: 50)

    Returns:
        Generated title
    """
    if not transcript or not transcript.strip():
        return "Untitled Recording"

    # Clean up the transcript
    text = transcript.strip()
    # Replace multiple whitespace/newlines with single space
    text = re.sub(r"\s+", " ", text)

    # Find first sentence (ending with . ! ? or end of text)
    sentence_match = re.match(r"^([^.!?]+[.!?]?)", text)
    first_sentence = sentence_match.group(1).strip() if sentence_match else text

    # Remove trailing punctuation for cleaner title
    first_sentence = first_sentence.rstrip(".!?")

    # Truncate to max_length
    if len(first_sentence) > max_length:
        # Try to break at word boundary
        truncated = first_sentence[:max_length].rsplit(" ", 1)[0]
        # If that results in too short, just hard truncate
        if len(truncated) < max_length * 0.7:
            truncated = first_sentence[:max_length]
        title = truncated + "..."
    else:
        title = first_sentence

    # Fallback if title is too short or empty
    if len(title.strip()) < 3:
        return "Untitled Recording"

    return title
