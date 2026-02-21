"""Chat service — orchestrates agentic tool-calling loop with streaming."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from harkd.chat.tool_executor import ToolExecutor
from harkd.chat.tools import CHAT_TOOLS
from harkd.llm.client import LLMClient
from harkd.storage.base import ChatThreadStorage
from harkd.storage.models import StorageChatMessage, StorageChatScope, StorageChatThread

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 5
SUMMARIZE_THRESHOLD = 40


@dataclass
class ChatStreamEvent:
    """SSE event yielded by ChatService."""

    type: Literal[
        "thread_created",
        "message_start",
        "token",
        "tool_call",
        "tool_result",
        "message_end",
        "error",
        "done",
    ]
    data: dict[str, Any]


class ChatService:
    """Orchestrates chat with agentic tool-calling loop and streaming."""

    def __init__(
        self,
        thread_storage: ChatThreadStorage,
        llm_client: LLMClient,
        tool_executor: ToolExecutor,
    ):
        self._threads = thread_storage
        self._llm = llm_client
        self._tools = tool_executor

    async def send_message(
        self,
        content: str,
        *,
        thread_id: str | None = None,
        scope: StorageChatScope | None = None,
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        """Send a user message and stream the assistant response.

        Args:
            content: User message text
            thread_id: Existing thread ID, or None to create new
            scope: Optional scope filters

        Yields:
            ChatStreamEvent instances for SSE
        """
        try:
            # 1. Create or retrieve thread
            thread, is_new = await self._get_or_create_thread(thread_id, scope)
            if is_new:
                yield ChatStreamEvent(
                    type="thread_created",
                    data={"thread_id": thread.id, "title": thread.title},
                )

            # 2. Append user message
            user_msg = StorageChatMessage(
                id=str(uuid.uuid4()),
                role="user",
                content=content,
                timestamp=datetime.now(UTC),
            )
            thread.messages.append(user_msg)

            # Update title from first user message
            if len([m for m in thread.messages if m.role == "user"]) == 1:
                thread.title = content[:60].strip()

            # 3. Build LLM context
            llm_messages = self._build_llm_messages(thread)

            # 4. Agentic tool-calling loop
            assistant_msg_id = str(uuid.uuid4())
            yield ChatStreamEvent(
                type="message_start",
                data={"message_id": assistant_msg_id},
            )

            full_content = ""
            all_tool_calls: list = []
            all_citations: list = []

            for _iteration in range(MAX_TOOL_ITERATIONS):
                # Stream LLM response
                tool_calls_in_round: dict[int, dict[str, Any]] = {}
                text_in_round = ""

                async for event in self._llm.astream(llm_messages, tools=CHAT_TOOLS):
                    if event.type == "token":
                        text_in_round += event.content
                        yield ChatStreamEvent(
                            type="token",
                            data={"content": event.content},
                        )
                    elif event.type == "tool_call_start":
                        idx = event.tool_call_index or 0
                        tool_calls_in_round[idx] = {
                            "id": event.tool_call_id or "",
                            "name": event.tool_name or "",
                            "args": "",
                        }
                        yield ChatStreamEvent(
                            type="tool_call",
                            data={
                                "tool_call_id": event.tool_call_id,
                                "tool_name": event.tool_name,
                                "status": "started",
                            },
                        )
                    elif event.type == "tool_call_args":
                        idx = event.tool_call_index or 0
                        if idx in tool_calls_in_round:
                            tool_calls_in_round[idx]["args"] += event.content
                    elif event.type == "tool_call_end":
                        idx = event.tool_call_index or 0
                        if idx in tool_calls_in_round:
                            tool_calls_in_round[idx]["args"] = event.content

                # If no tool calls, we're done
                if not tool_calls_in_round:
                    full_content += text_in_round
                    break

                # Process tool calls
                # First, append the assistant message with tool_calls to context
                parsed_tool_calls = []
                for tc in tool_calls_in_round.values():
                    try:
                        args = json.loads(tc["args"]) if tc["args"] else {}
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool call args: {tc['args'][:200]}")
                        args = {}
                    parsed_tool_calls.append(
                        {
                            "id": tc["id"],
                            "name": tc["name"],
                            "args": args,
                        }
                    )

                all_tool_calls.extend(parsed_tool_calls)

                llm_messages.append(
                    {
                        "role": "assistant",
                        "content": text_in_round,
                        "tool_calls": parsed_tool_calls,
                    }
                )

                # Execute each tool call
                for tc in parsed_tool_calls:
                    result = await self._tools.execute(tc["name"], tc["args"], scope=thread.scope)

                    # Truncate very large results
                    if len(result) > 8000:
                        logger.debug(f"Truncated tool result from {len(result)} to 8000 chars")
                        result = result[:8000] + "\n... (truncated)"

                    llm_messages.append(
                        {
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tc["id"],
                        }
                    )

                    # Append tool message to thread
                    thread.messages.append(
                        StorageChatMessage(
                            id=str(uuid.uuid4()),
                            role="tool",
                            content=result,
                            timestamp=datetime.now(UTC),
                            tool_call_id=tc["id"],
                            tool_name=tc["name"],
                        )
                    )

                    yield ChatStreamEvent(
                        type="tool_result",
                        data={
                            "tool_call_id": tc["id"],
                            "tool_name": tc["name"],
                            "result_preview": result[:200],
                        },
                    )

                full_content += text_in_round

            # 5. Append assistant message to thread
            assistant_msg = StorageChatMessage(
                id=assistant_msg_id,
                role="assistant",
                content=full_content,
                timestamp=datetime.now(UTC),
                tool_calls=all_tool_calls,
                citations=all_citations,
            )
            thread.messages.append(assistant_msg)

            yield ChatStreamEvent(
                type="message_end",
                data={"message_id": assistant_msg_id},
            )

            # 6. Persist thread
            thread.updated_at = datetime.now(UTC)
            try:
                if is_new:
                    await self._threads.create(thread)
                else:
                    await self._threads.update(thread)
            except Exception:
                logger.error("Failed to persist chat thread", exc_info=True)

            # 7. Trigger context summarization if needed
            if len(thread.messages) > SUMMARIZE_THRESHOLD:
                await self._maybe_summarize(thread)

            yield ChatStreamEvent(type="done", data={})

        except Exception as e:
            logger.error(f"Chat service error: {e}", exc_info=True)
            yield ChatStreamEvent(
                type="error",
                data={"message": str(e)},
            )

    async def _get_or_create_thread(
        self,
        thread_id: str | None,
        scope: StorageChatScope | None,
    ) -> tuple[StorageChatThread, bool]:
        """Get existing thread or create a new one."""
        if thread_id:
            thread = await self._threads.get(thread_id)
            if thread:
                # Update scope if provided
                if scope:
                    thread.scope = scope
                return thread, False

        # Create new thread
        now = datetime.now(UTC)
        thread = StorageChatThread(
            id=str(uuid.uuid4()),
            title="New conversation",
            created_at=now,
            updated_at=now,
            scope=scope or StorageChatScope(),
        )
        return thread, True

    def _build_llm_messages(self, thread: StorageChatThread) -> list[dict[str, Any]]:
        """Build LLM context from thread state."""
        # System prompt
        scope_context = ""
        if thread.scope.recording_ids:
            rec_ids = ", ".join(thread.scope.recording_ids)
            scope_context = (
                "\n\n## Current scope\n"
                f"This conversation is scoped to recording(s): "
                f"{rec_ids}. Focus your tool calls on these "
                "recordings unless the user asks to broaden."
            )
        elif thread.scope.speakers or thread.scope.tags:
            parts = []
            if thread.scope.speakers:
                parts.append(f"speakers: {', '.join(thread.scope.speakers)}")
            if thread.scope.tags:
                parts.append(f"tags: {', '.join(thread.scope.tags)}")
            scope_context = f"\n\n## Current scope\nFiltered by {'; '.join(parts)}."

        system_prompt = self._llm.prompts.get("chat_system").format(
            current_date=datetime.now(UTC).strftime("%Y-%m-%d"),
            scope_context=scope_context,
        )

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        # Add context summary if present
        if thread.context_summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Summary of earlier conversation:\n{thread.context_summary}",
                }
            )

        # Add recent messages (last N)
        recent_count = thread.recent_message_count
        recent_msgs = thread.messages[-recent_count:]

        for msg in recent_msgs:
            if msg.role == "user":
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content,
                }
                if msg.tool_calls:
                    entry["tool_calls"] = msg.tool_calls
                messages.append(entry)
            elif msg.role == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id or "",
                    }
                )

        return messages

    async def _maybe_summarize(self, thread: StorageChatThread) -> None:
        """Summarize older messages to keep context manageable."""
        try:
            recent_count = thread.recent_message_count
            if len(thread.messages) <= recent_count:
                return

            # Messages to summarize
            older = thread.messages[:-recent_count]
            conversation = "\n".join(
                f"{m.role}: {m.content[:500]}" for m in older if m.role in ("user", "assistant")
            )

            if not conversation.strip():
                return

            prompt = self._llm.prompts.get("chat_summarize").format(conversation=conversation)

            response = await self._llm.invoke([{"role": "user", "content": prompt}])

            thread.context_summary = response.content
            # Keep only recent messages
            thread.messages = thread.messages[-recent_count:]
            thread.updated_at = datetime.now(UTC)
            await self._threads.update(thread)

        except Exception:
            logger.warning("Failed to summarize chat context", exc_info=True)
