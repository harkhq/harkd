"""Tests for title generation."""

from harkd.services.title_generator import generate_title


def test_generate_title_from_simple_sentence():
    """Test generating title from simple sentence."""
    transcript = "Hello everyone, this is a test recording."
    title = generate_title(transcript)
    assert title == "Hello everyone, this is a test recording"


def test_generate_title_removes_trailing_punctuation():
    """Test that trailing punctuation is removed."""
    transcript = "This is a meeting."
    title = generate_title(transcript)
    assert title == "This is a meeting"


def test_generate_title_truncates_long_text():
    """Test that long text is truncated."""
    transcript = (
        "This is a very long sentence that goes on and on"
        " and should be truncated at some point."
    )
    title = generate_title(transcript, max_length=30)
    assert len(title) <= 33  # 30 + "..."
    assert title.endswith("...")


def test_generate_title_first_sentence_only():
    """Test that only first sentence is used."""
    transcript = "First sentence here. Second sentence should not appear."
    title = generate_title(transcript)
    assert title == "First sentence here"
    assert "Second" not in title


def test_generate_title_handles_question():
    """Test that questions are handled correctly."""
    transcript = "How are you doing? I'm doing great."
    title = generate_title(transcript)
    assert title == "How are you doing"


def test_generate_title_handles_exclamation():
    """Test that exclamations are handled correctly."""
    transcript = "This is amazing! Really cool stuff."
    title = generate_title(transcript)
    assert title == "This is amazing"


def test_generate_title_empty_transcript():
    """Test handling of empty transcript."""
    title = generate_title("")
    assert title == "Untitled Recording"


def test_generate_title_whitespace_only():
    """Test handling of whitespace-only transcript."""
    title = generate_title("   \n\n  \t  ")
    assert title == "Untitled Recording"


def test_generate_title_normalizes_whitespace():
    """Test that multiple whitespace is normalized."""
    transcript = "Hello    everyone,\n\nthis   is    a  test."
    title = generate_title(transcript)
    assert title == "Hello everyone, this is a test"
    assert "  " not in title
    assert "\n" not in title


def test_generate_title_no_sentence_ending():
    """Test handling of text without sentence ending."""
    transcript = "Just some text without punctuation"
    title = generate_title(transcript)
    assert title == "Just some text without punctuation"


def test_generate_title_truncates_at_word_boundary():
    """Test that truncation happens at word boundary."""
    transcript = "The quick brown fox jumps over the lazy dog every single day"
    title = generate_title(transcript, max_length=30)
    # Should truncate at word boundary
    assert title.endswith("...")
    assert not title.endswith(" ...")  # No trailing space before ...


def test_generate_title_very_short_text():
    """Test handling of very short text."""
    title = generate_title("Hi there")
    assert title == "Hi there"


def test_generate_title_single_char():
    """Test handling of single character (too short)."""
    title = generate_title("A")
    assert title == "Untitled Recording"


def test_generate_title_custom_max_length():
    """Test custom max length."""
    transcript = "This is a test of custom maximum length settings."
    title = generate_title(transcript, max_length=20)
    assert len(title) <= 23  # 20 + "..."
    assert title.endswith("...")


def test_generate_title_multiline_transcript():
    """Test handling of multiline transcript."""
    transcript = """Line one here.
Line two here.
Line three here."""
    title = generate_title(transcript)
    assert title == "Line one here"
