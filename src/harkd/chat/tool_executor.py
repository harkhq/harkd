"""Execute chat tools against the recording storage layer."""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime
from typing import Any

from harkd.chat.tools import (
    GetDecisionsInput,
    GetRecordingDetailInput,
    GetTasksInput,
    GetTranscriptSegmentsInput,
    SearchRecordingsInput,
)
from harkd.storage.base import RecordingStorage
from harkd.storage.models import StorageChatScope, StorageRecording

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes chat tool calls against recording storage."""

    def __init__(self, storage: RecordingStorage):
        self._storage = storage

    async def execute(
        self, tool_name: str, tool_args: dict[str, Any], scope: StorageChatScope | None = None
    ) -> str:
        """Execute a tool call and return JSON string result."""
        try:
            match tool_name:
                case "SearchRecordingsInput":
                    args = SearchRecordingsInput.model_validate(tool_args)
                    result = await self._search_recordings(args, scope)
                case "GetRecordingDetailInput":
                    args = GetRecordingDetailInput.model_validate(tool_args)
                    result = await self._get_recording_detail(args, scope)
                case "GetTranscriptSegmentsInput":
                    args = GetTranscriptSegmentsInput.model_validate(tool_args)
                    result = await self._get_transcript_segments(args, scope)
                case "GetTasksInput":
                    args = GetTasksInput.model_validate(tool_args)
                    result = await self._get_tasks(args, scope)
                case "GetDecisionsInput":
                    args = GetDecisionsInput.model_validate(tool_args)
                    result = await self._get_decisions(args, scope)
                case _:
                    result = {"error": f"Unknown tool: {tool_name}"}
            return json.dumps(result, default=str, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return json.dumps({"error": str(e)})

    async def _get_completed_recordings(
        self, scope: StorageChatScope | None = None
    ) -> list[StorageRecording]:
        """Get all completed recordings, filtered by scope."""
        recordings = await self._storage.list(limit=0, status="complete")
        if scope:
            recordings = self._apply_scope(recordings, scope)
        return recordings

    @staticmethod
    def _apply_scope(
        recordings: list[StorageRecording], scope: StorageChatScope
    ) -> list[StorageRecording]:
        """Apply scope filters to recordings."""
        result = recordings

        if scope.recording_ids:
            ids = set(scope.recording_ids)
            result = [r for r in result if r.id in ids]

        if scope.date_from:
            if isinstance(scope.date_from, datetime):
                dt_from = scope.date_from
            else:
                dt_from = datetime.combine(scope.date_from, datetime.min.time(), tzinfo=UTC)
            result = [r for r in result if r.created_at >= dt_from]

        if scope.date_to:
            if isinstance(scope.date_to, datetime):
                dt_to = scope.date_to
            else:
                dt_to = datetime.combine(scope.date_to, datetime.max.time(), tzinfo=UTC)
            result = [r for r in result if r.created_at <= dt_to]

        if scope.speakers:
            scope_speakers = {s.lower() for s in scope.speakers}
            result = [r for r in result if any(s.lower() in scope_speakers for s in r.speakers)]

        if scope.tags:
            scope_tags = {t.lower() for t in scope.tags}
            result = [r for r in result if any(t.lower() in scope_tags for t in r.tags)]

        return result

    @staticmethod
    def _matches_date_filter(
        recording: StorageRecording,
        date_from: date | None,
        date_to: date | None,
    ) -> bool:
        """Check if recording falls within date range."""
        if isinstance(recording.created_at, datetime):
            rec_date = recording.created_at.date()
        else:
            rec_date = recording.created_at
        if date_from and rec_date < date_from:
            return False
        return not (date_to and rec_date > date_to)

    async def _search_recordings(
        self, args: SearchRecordingsInput, scope: StorageChatScope | None
    ) -> list[dict[str, Any]]:
        recordings = await self._get_completed_recordings(scope)

        # Apply tool-level filters
        if args.date_from or args.date_to:
            recordings = [
                r for r in recordings if self._matches_date_filter(r, args.date_from, args.date_to)
            ]

        if args.speakers:
            filter_speakers = {s.lower() for s in args.speakers}
            recordings = [
                r for r in recordings if any(s.lower() in filter_speakers for s in r.speakers)
            ]

        if args.tags:
            filter_tags = {t.lower() for t in args.tags}
            recordings = [r for r in recordings if any(t.lower() in filter_tags for t in r.tags)]

        # Keyword search on title + transcript
        if args.query:
            query_lower = args.query.lower()
            recordings = [
                r
                for r in recordings
                if query_lower in r.title.lower()
                or (r.transcript and query_lower in r.transcript.lower())
            ]

        # Sort by date descending
        recordings.sort(key=lambda r: r.created_at, reverse=True)

        return [
            {
                "id": r.id,
                "title": r.title,
                "date": r.created_at.isoformat(),
                "speakers": r.speakers,
                "executive_summary": r.executive_summary,
            }
            for r in recordings[: args.limit]
        ]

    async def _get_recording_detail(
        self, args: GetRecordingDetailInput, scope: StorageChatScope | None
    ) -> dict[str, Any]:
        recording = await self._storage.get(args.recording_id)
        if not recording or recording.status != "complete":
            return {"error": f"Recording {args.recording_id} not found or not complete"}

        # Verify scope access
        if scope and scope.recording_ids and args.recording_id not in scope.recording_ids:
            return {"error": f"Recording {args.recording_id} is outside the current scope"}

        transcript_preview = (recording.transcript or "")[:3000]

        return {
            "id": recording.id,
            "title": recording.title,
            "date": recording.created_at.isoformat(),
            "duration": recording.duration,
            "speakers": recording.speakers,
            "language": recording.language,
            "tags": recording.tags,
            "executive_summary": recording.executive_summary,
            "meeting_notes": recording.meeting_notes,
            "tasks": recording.tasks,
            "decisions": recording.decisions,
            "transcript_preview": transcript_preview,
        }

    async def _get_transcript_segments(
        self, args: GetTranscriptSegmentsInput, scope: StorageChatScope | None
    ) -> list[dict[str, Any]]:
        recordings = await self._get_completed_recordings(scope)

        if args.recording_ids:
            ids = set(args.recording_ids)
            recordings = [r for r in recordings if r.id in ids]

        query_lower = args.query.lower()
        matches: list[dict[str, Any]] = []

        for recording in recordings:
            segments = recording.segments
            for i, seg in enumerate(segments):
                text = seg.get("text", "")
                if query_lower not in text.lower():
                    continue

                # Include +/- 1 context segment
                context_segments = []
                if i > 0:
                    context_segments.append(segments[i - 1])
                context_segments.append(seg)
                if i < len(segments) - 1:
                    context_segments.append(segments[i + 1])

                matches.append(
                    {
                        "recording_id": recording.id,
                        "recording_title": recording.title,
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "speaker": seg.get("speaker"),
                        "text": text,
                        "context": [
                            {
                                "speaker": s.get("speaker"),
                                "text": s.get("text", ""),
                                "start": s.get("start"),
                                "end": s.get("end"),
                            }
                            for s in context_segments
                        ],
                    }
                )

                if len(matches) >= args.limit:
                    return matches

        return matches

    async def _get_tasks(
        self, args: GetTasksInput, scope: StorageChatScope | None
    ) -> list[dict[str, Any]]:
        recordings = await self._get_completed_recordings(scope)

        if args.recording_ids:
            ids = set(args.recording_ids)
            recordings = [r for r in recordings if r.id in ids]

        if args.date_from or args.date_to:
            recordings = [
                r for r in recordings if self._matches_date_filter(r, args.date_from, args.date_to)
            ]

        tasks: list[dict[str, Any]] = []
        for recording in recordings:
            for task in recording.tasks:
                if args.assignee:
                    assignee = task.get("assignee", "")
                    if assignee and args.assignee.lower() not in assignee.lower():
                        continue

                tasks.append(
                    {
                        "recording_id": recording.id,
                        "recording_title": recording.title,
                        "recording_date": recording.created_at.isoformat(),
                        **task,
                    }
                )

                if len(tasks) >= 50:
                    return tasks

        return tasks

    async def _get_decisions(
        self, args: GetDecisionsInput, scope: StorageChatScope | None
    ) -> list[dict[str, Any]]:
        recordings = await self._get_completed_recordings(scope)

        if args.recording_ids:
            ids = set(args.recording_ids)
            recordings = [r for r in recordings if r.id in ids]

        if args.date_from or args.date_to:
            recordings = [
                r for r in recordings if self._matches_date_filter(r, args.date_from, args.date_to)
            ]

        decisions: list[dict[str, Any]] = []
        for recording in recordings:
            for decision in recording.decisions:
                decisions.append(
                    {
                        "recording_id": recording.id,
                        "recording_title": recording.title,
                        "recording_date": recording.created_at.isoformat(),
                        "decision": decision,
                    }
                )

                if len(decisions) >= 50:
                    return decisions

        return decisions
