"""Tool input schemas for LangChain bind_tools()."""

from datetime import date

from pydantic import BaseModel, Field


class SearchRecordingsInput(BaseModel):
    """Search recordings by keyword, date range, speakers, or tags."""

    query: str | None = Field(default=None, description="Search query for title or transcript")
    date_from: date | None = Field(default=None, description="Filter: on or after this date")
    date_to: date | None = Field(default=None, description="Filter: on or before this date")
    speakers: list[str] = Field(default_factory=list, description="Filter by speaker names")
    tags: list[str] = Field(default_factory=list, description="Filter by tags")
    limit: int = Field(default=10, ge=1, le=10, description="Max results")


class GetRecordingDetailInput(BaseModel):
    """Get full details for a specific recording."""

    recording_id: str = Field(description="The recording ID to retrieve")


class GetTranscriptSegmentsInput(BaseModel):
    """Search transcript segments across recordings by keyword."""

    query: str = Field(description="Keyword or phrase to search in transcripts")
    recording_ids: list[str] = Field(
        default_factory=list,
        description="Limit search to these recording IDs",
    )
    limit: int = Field(default=20, ge=1, le=20, description="Max matching segments")


class GetTasksInput(BaseModel):
    """Get action items / tasks extracted from meetings."""

    recording_ids: list[str] = Field(
        default_factory=list, description="Limit to these recording IDs"
    )
    assignee: str | None = Field(default=None, description="Filter by assignee name")
    date_from: date | None = Field(default=None, description="Filter: on or after this date")
    date_to: date | None = Field(default=None, description="Filter: on or before this date")


class GetDecisionsInput(BaseModel):
    """Get decisions made in meetings."""

    recording_ids: list[str] = Field(
        default_factory=list, description="Limit to these recording IDs"
    )
    date_from: date | None = Field(default=None, description="Filter: on or after this date")
    date_to: date | None = Field(default=None, description="Filter: on or before this date")


CHAT_TOOLS = [
    SearchRecordingsInput,
    GetRecordingDetailInput,
    GetTranscriptSegmentsInput,
    GetTasksInput,
    GetDecisionsInput,
]
