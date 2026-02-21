"""Main LLM client — entry point for all LLM interactions."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
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
from harkd.llm.types import LLMResponse, LLMStreamEvent, MeetingMinutesResult, TokenUsage

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
    def prompts(self) -> PromptManager:
        """Access the prompt manager."""
        return self._prompts

    @property
    def token_stats(self) -> TokenUsage | None:
        """Get cumulative token usage stats, if tracking is enabled."""
        if self._token_stats:
            return self._token_stats.total_usage
        return None

    @staticmethod
    def _build_lc_messages(messages: list[dict[str, Any]]) -> list[Any]:
        """Convert dict messages to LangChain message objects.

        Supports system, user (human), assistant (with optional tool_calls),
        and tool message roles.
        """
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    lc_messages.append(AIMessage(content=content, tool_calls=tool_calls))
                else:
                    lc_messages.append(AIMessage(content=content))
            elif role == "tool":
                lc_messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=msg.get("tool_call_id", ""),
                    )
                )
            else:
                lc_messages.append(HumanMessage(content=content))
        return lc_messages

    async def _call_model(self, request: dict[str, Any]) -> LLMResponse:
        """Execute the actual LLM call via LangChain.

        Args:
            request: Dict with 'messages' and optional 'tools'

        Returns:
            LLMResponse with content and usage
        """
        messages = self._build_lc_messages(request.get("messages", []))

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

    async def astream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list | None = None,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        """Stream LLM response token-by-token.

        Bypasses middleware chain (streaming is incompatible with caching).
        Converts LangChain AIMessageChunk events to LLMStreamEvent.

        Args:
            messages: List of message dicts
            tools: Optional tool schemas for bind_tools

        Yields:
            LLMStreamEvent for each chunk
        """
        lc_messages = self._build_lc_messages(messages)

        model = self._model
        if tools:
            model = model.bind_tools(tools)

        # Track tool call state across chunks
        active_tool_calls: dict[int, dict[str, Any]] = {}

        async for chunk in model.astream(lc_messages):
            # Text content
            if chunk.content:
                text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                if text:
                    yield LLMStreamEvent(type="token", content=text)

            # Tool calls from chunk
            tool_call_chunks = getattr(chunk, "tool_call_chunks", [])
            for tc_chunk in tool_call_chunks:
                idx = tc_chunk.get("index", 0)
                name = tc_chunk.get("name")
                args_str = tc_chunk.get("args", "")
                tc_id = tc_chunk.get("id")

                if name and idx not in active_tool_calls:
                    # New tool call starting
                    active_tool_calls[idx] = {
                        "id": tc_id or "",
                        "name": name,
                        "args": "",
                    }
                    yield LLMStreamEvent(
                        type="tool_call_start",
                        tool_call_id=tc_id or "",
                        tool_name=name,
                        tool_call_index=idx,
                    )

                if args_str and idx in active_tool_calls:
                    active_tool_calls[idx]["args"] += args_str
                    yield LLMStreamEvent(
                        type="tool_call_args",
                        content=args_str,
                        tool_call_index=idx,
                    )

            # Check for tool_calls on the chunk (non-streaming tool calls)
            tool_calls = getattr(chunk, "tool_calls", [])
            for i, tc in enumerate(tool_calls):
                if i not in active_tool_calls:
                    tc_id = tc.get("id", "")
                    tc_name = tc.get("name", "")
                    tc_args = json.dumps(tc.get("args", {}))
                    active_tool_calls[i] = {
                        "id": tc_id,
                        "name": tc_name,
                        "args": tc_args,
                    }
                    yield LLMStreamEvent(
                        type="tool_call_start",
                        tool_call_id=tc_id,
                        tool_name=tc_name,
                        tool_call_index=i,
                    )
                    yield LLMStreamEvent(
                        type="tool_call_args",
                        content=tc_args,
                        tool_call_index=i,
                    )

            # Usage info
            usage_metadata = getattr(chunk, "usage_metadata", None)
            if usage_metadata:
                yield LLMStreamEvent(
                    type="usage",
                    usage=TokenUsage(
                        prompt_tokens=usage_metadata.get("input_tokens", 0),
                        completion_tokens=usage_metadata.get("output_tokens", 0),
                        total_tokens=usage_metadata.get("total_tokens", 0),
                    ),
                )

        # Emit tool_call_end for all active tool calls
        for idx, tc in active_tool_calls.items():
            yield LLMStreamEvent(
                type="tool_call_end",
                tool_call_id=tc["id"],
                tool_name=tc["name"],
                content=tc["args"],
                tool_call_index=idx,
            )

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
