"""REST API routes for chat (Ask Hark)."""

from __future__ import annotations

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse

from harkd.api.deps import get_chat_service
from harkd.api.models.chat import (
    ChatMessageResponse,
    ChatScopeModel,
    ChatSendRequest,
    ChatThreadListItem,
    ChatThreadListResponse,
    ChatThreadResponse,
    ChatThreadUpdate,
    CitationModel,
)
from harkd.chat.service import ChatService
from harkd.exceptions import ChatThreadNotFoundError
from harkd.storage.models import StorageChatScope, StorageChatThread

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


def _scope_to_storage(scope: ChatScopeModel | None) -> StorageChatScope | None:
    """Convert API scope model to storage model."""
    if scope is None:
        return None
    return StorageChatScope.model_validate(scope.model_dump())


def _scope_to_api(scope: StorageChatScope) -> ChatScopeModel:
    """Convert storage scope model to API model."""
    return ChatScopeModel.model_validate(scope.model_dump())


def _thread_to_response(
    thread: StorageChatThread,
    *,
    include_messages: bool = True,
) -> ChatThreadResponse:
    """Convert storage thread to API response."""
    messages = []
    if include_messages:
        for m in thread.messages:
            if m.role not in ("user", "assistant"):
                continue
            citations = [CitationModel.model_validate(c) for c in m.citations]
            messages.append(
                ChatMessageResponse(
                    id=m.id,
                    role=m.role,
                    content=m.content,
                    timestamp=m.timestamp,
                    tool_call_id=m.tool_call_id,
                    tool_name=m.tool_name,
                    citations=citations,
                )
            )

    return ChatThreadResponse(
        id=thread.id,
        title=thread.title,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
        scope=_scope_to_api(thread.scope),
        messages=messages,
    )


@router.post(
    "/send",
    summary="Send chat message (SSE stream)",
    description=("Send a message and receive a streaming response via Server-Sent Events."),
)
async def send_message(
    request: ChatSendRequest,
    service: ChatServiceDep,
) -> StreamingResponse:
    """Send a message and stream the response."""

    async def event_stream():
        async for event in service.send_message(
            content=request.message,
            thread_id=request.thread_id,
            scope=_scope_to_storage(request.scope),
        ):
            data = json.dumps(event.data, default=str, ensure_ascii=False)
            yield f"event: {event.type}\ndata: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/threads",
    response_model=ChatThreadListResponse,
    summary="List chat threads",
)
async def list_threads(
    service: ChatServiceDep,
) -> ChatThreadListResponse:
    """List all chat threads."""
    threads = await service._threads.list()
    items = [
        ChatThreadListItem(
            id=t.id,
            title=t.title,
            created_at=t.created_at,
            updated_at=t.updated_at,
            scope=_scope_to_api(t.scope),
        )
        for t in threads
    ]
    return ChatThreadListResponse(threads=items, total=len(items))


@router.get(
    "/threads/{thread_id}",
    response_model=ChatThreadResponse,
    summary="Get chat thread",
)
async def get_thread(
    thread_id: str,
    service: ChatServiceDep,
) -> ChatThreadResponse:
    """Get a chat thread by ID."""
    thread = await service._threads.get(thread_id)
    if not thread:
        raise ChatThreadNotFoundError(thread_id)
    return _thread_to_response(thread)


@router.patch(
    "/threads/{thread_id}",
    response_model=ChatThreadResponse,
    summary="Update chat thread",
)
async def update_thread(
    thread_id: str,
    update: ChatThreadUpdate,
    service: ChatServiceDep,
) -> ChatThreadResponse:
    """Update a chat thread (title/scope)."""
    thread = await service._threads.get(thread_id)
    if not thread:
        raise ChatThreadNotFoundError(thread_id)

    if update.title is not None:
        thread.title = update.title
    if update.scope is not None:
        thread.scope = StorageChatScope.model_validate(update.scope.model_dump())

    await service._threads.update(thread)
    return _thread_to_response(thread)


@router.delete(
    "/threads/{thread_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete chat thread",
)
async def delete_thread(
    thread_id: str,
    service: ChatServiceDep,
) -> None:
    """Delete a chat thread."""
    await service._threads.delete(thread_id)
