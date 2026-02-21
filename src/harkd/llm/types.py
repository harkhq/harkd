"""Shared types for the LLM abstraction layer."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class TokenUsage:
    """Token usage statistics from an LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    usage: TokenUsage | None = None
    model: str | None = None
    cached: bool = False


@dataclass
class MeetingMinutesResult:
    """Structured meeting minutes extracted from a transcript."""

    executive_summary: list[str] = field(default_factory=list)
    meeting_notes: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)


@dataclass
class LLMStreamEvent:
    """A single event from an LLM streaming response."""

    type: Literal["token", "tool_call_start", "tool_call_args", "tool_call_end", "usage", "error"]
    content: str = ""
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_call_index: int | None = None
    usage: TokenUsage | None = None
