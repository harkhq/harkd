"""Main LLM client — entry point for all LLM interactions."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from harkd.llm.middleware import (
    CacheMiddleware,
    LoggingMiddleware,
    Middleware,
    TokenStatsMiddleware,
    build_middleware_chain,
)
from harkd.llm.prompts import PromptManager
from harkd.llm.providers import create_chat_model
from harkd.llm.types import LLMResponse, MeetingMinutesResult, TokenUsage

if TYPE_CHECKING:
    from harkd.config import LLMSettings

logger = logging.getLogger(__name__)


class LLMClient:
    """High-level LLM client with middleware chain and prompt management."""

    def __init__(
        self,
        config: LLMSettings,
        prompt_manager: PromptManager | None = None,
    ):
        """Initialize LLM client.

        Args:
            config: LLM configuration settings
            prompt_manager: Optional custom prompt manager
        """
        self._config = config
        self._model = create_chat_model(config)
        self._prompts = prompt_manager or PromptManager(config.custom_prompts_dir)

        # Build middleware chain
        middlewares: list[Middleware] = []
        if config.enable_cache:
            middlewares.append(CacheMiddleware())
        if config.enable_logging:
            middlewares.append(LoggingMiddleware())
        self._token_stats: TokenStatsMiddleware | None = None
        if config.enable_token_stats:
            self._token_stats = TokenStatsMiddleware()
            middlewares.append(self._token_stats)

        self._chain = build_middleware_chain(middlewares, self._call_model)

    @property
    def token_stats(self) -> TokenUsage | None:
        """Get cumulative token usage stats, if tracking is enabled."""
        if self._token_stats:
            return self._token_stats.total_usage
        return None

    async def _call_model(self, request: dict[str, Any]) -> LLMResponse:
        """Execute the actual LLM call via LangChain.

        Args:
            request: Dict with 'messages' and optional 'tools'

        Returns:
            LLMResponse with content and usage
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = []
        for msg in request.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                messages.append(SystemMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))

        # Bind tools if provided
        model = self._model
        tools = request.get("tools")
        if tools:
            model = model.bind_tools(tools)

        result = await model.ainvoke(messages)

        # Extract token usage from response metadata
        usage = None
        usage_metadata = getattr(result, "usage_metadata", None)
        if usage_metadata:
            usage = TokenUsage(
                prompt_tokens=usage_metadata.get("input_tokens", 0),
                completion_tokens=usage_metadata.get("output_tokens", 0),
                total_tokens=usage_metadata.get("total_tokens", 0),
            )

        return LLMResponse(
            content=result.content if isinstance(result.content, str) else str(result.content),
            usage=usage,
            model=self._config.model,
        )

    async def invoke(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Generic LLM call through middleware chain.

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            tools: Optional list of tool/function definitions

        Returns:
            LLM response
        """
        request: dict[str, Any] = {"messages": messages}
        if tools:
            request["tools"] = tools

        return await self._chain(request)

    async def invoke_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
    ) -> LLMResponse:
        """LLM call with tool/function definitions.

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            tools: List of tool/function definitions for bind_tools

        Returns:
            LLM response
        """
        return await self.invoke(messages, tools=tools)

    async def generate_meeting_minutes(
        self,
        transcript: str,
        speakers: list[str],
        language: str = "en",
    ) -> MeetingMinutesResult:
        """Generate structured meeting minutes from a transcript.

        Args:
            transcript: Full transcript text
            speakers: List of speaker labels/names
            language: Language code

        Returns:
            Structured meeting minutes
        """
        prompt_template = self._prompts.get("meeting_minutes")
        prompt = prompt_template.format(
            transcript=transcript,
            speakers=", ".join(speakers) if speakers else "Unknown",
            language=language,
        )

        response = await self.invoke(
            [{"role": "user", "content": prompt}],
        )

        return self._parse_meeting_minutes(response.content)

    @staticmethod
    def _parse_meeting_minutes(content: str) -> MeetingMinutesResult:
        """Parse LLM response into MeetingMinutesResult.

        Handles JSON wrapped in markdown code blocks.

        Args:
            content: Raw LLM response text

        Returns:
            Parsed meeting minutes result
        """
        # Strip markdown code fences if present
        text = content.strip()
        if text.startswith("```"):
            # Remove first line (```json or ```)
            lines = text.split("\n")
            # Find closing ```
            start = 1
            end = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == "```":
                    end = i
                    break
            text = "\n".join(lines[start:end])

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse meeting minutes JSON: {text[:200]}")
            return MeetingMinutesResult(
                executive_summary=["Meeting minutes could not be parsed."],
            )

        return MeetingMinutesResult(
            executive_summary=data.get("executive_summary", []),
            meeting_notes=data.get("meeting_notes", []),
            tasks=data.get("tasks", []),
            decisions=data.get("decisions", []),
        )
