"""API request/response models for chat."""

from datetime import datetime

from pydantic import BaseModel, Field


class ChatScopeModel(BaseModel):
    """Scope filters for a chat request."""

    recording_ids: list[str] = Field(default_factory=list)
    date_from: datetime | None = None
    date_to: datetime | None = None
    speakers: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class ChatSendRequest(BaseModel):
    """Request body for POST /api/v1/chat/send."""

    message: str = Field(min_length=1, max_length=10000, description="User message")
    thread_id: str | None = Field(default=None, description="Existing thread ID, or null for new")
    scope: ChatScopeModel | None = Field(default=None, description="Optional scope filters")


class CitationModel(BaseModel):
    """A citation referencing a recording."""

    recording_id: str
    recording_title: str | None = None
    timestamp: float | None = None


class ChatMessageResponse(BaseModel):
    """A single chat message in the API response."""

    id: str
    role: str
    content: str
    timestamp: datetime
    tool_call_id: str | None = None
    tool_name: str | None = None
    citations: list[CitationModel] = Field(default_factory=list)


class ChatThreadResponse(BaseModel):
    """Full chat thread response."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    scope: ChatScopeModel
    messages: list[ChatMessageResponse] = Field(default_factory=list)


class ChatThreadListItem(BaseModel):
    """Lightweight thread entry for list responses."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    scope: ChatScopeModel


class ChatThreadListResponse(BaseModel):
    """Response for GET /api/v1/chat/threads."""

    threads: list[ChatThreadListItem] = Field(default_factory=list)
    total: int = 0


class ChatThreadUpdate(BaseModel):
    """Request body for PATCH /api/v1/chat/threads/{id}."""

    title: str | None = None
    scope: ChatScopeModel | None = None
