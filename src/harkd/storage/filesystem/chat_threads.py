"""Filesystem-based chat thread storage implementation."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path

from harkd.exceptions import ChatThreadNotFoundError, StorageError
from harkd.storage.base import ChatThreadStorage
from harkd.storage.models import StorageChatThread

__all__ = ["FilesystemChatThreadStorage"]

logger = logging.getLogger(__name__)


class FilesystemChatThreadStorage(ChatThreadStorage):
    """
    Filesystem-based chat thread storage.

    Storage structure:
        {base_path}/chat-threads/
            {thread-id}.json
    """

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.threads_dir = base_path / "chat-threads"

    async def create(self, thread: StorageChatThread) -> StorageChatThread:
        """Create a new chat thread."""
        await asyncio.to_thread(self.threads_dir.mkdir, parents=True, exist_ok=True)

        thread_file = self.threads_dir / f"{thread.id}.json"

        if await asyncio.to_thread(thread_file.exists):
            raise StorageError(
                f"Chat thread {thread.id} already exists",
                details={"thread_id": thread.id},
            )

        try:
            await self._write_thread(thread_file, thread)
            return thread
        except Exception as e:
            await asyncio.to_thread(thread_file.unlink, missing_ok=True)
            raise StorageError(
                f"Failed to create chat thread: {e}",
                details={"thread_id": thread.id},
            ) from e

    async def get(self, thread_id: str) -> StorageChatThread | None:
        """Get a chat thread by ID."""
        thread_file = self.threads_dir / f"{thread_id}.json"

        if not await asyncio.to_thread(thread_file.exists):
            return None

        try:
            return await self._read_thread(thread_file)
        except Exception as e:
            raise StorageError(
                f"Failed to read chat thread {thread_id}: {e}",
                details={"thread_id": thread_id},
            ) from e

    async def list(self) -> list[StorageChatThread]:
        """List all chat threads (metadata only, messages stripped)."""
        try:
            if not await asyncio.to_thread(self.threads_dir.exists):
                return []

            thread_files = await asyncio.to_thread(lambda: list(self.threads_dir.glob("*.json")))

            threads = []
            for thread_file in thread_files:
                try:
                    thread = await self._read_thread(thread_file)
                    # Strip messages for list performance
                    thread.messages = []
                    thread.context_summary = None
                    threads.append(thread)
                except Exception:
                    logger.warning(f"Skipping corrupted chat thread: {thread_file}")
                    continue

            # Sort by updated_at descending
            threads.sort(key=lambda t: t.updated_at, reverse=True)
            return threads

        except Exception as e:
            raise StorageError(f"Failed to list chat threads: {e}") from e

    async def update(self, thread: StorageChatThread) -> StorageChatThread:
        """Update an existing chat thread."""
        thread_file = self.threads_dir / f"{thread.id}.json"

        if not await asyncio.to_thread(thread_file.exists):
            raise ChatThreadNotFoundError(thread.id)

        try:
            await self._write_thread(thread_file, thread)
            return thread
        except Exception as e:
            raise StorageError(
                f"Failed to update chat thread {thread.id}: {e}",
                details={"thread_id": thread.id},
            ) from e

    async def delete(self, thread_id: str) -> None:
        """Delete a chat thread."""
        thread_file = self.threads_dir / f"{thread_id}.json"

        if not await asyncio.to_thread(thread_file.exists):
            raise ChatThreadNotFoundError(thread_id)

        try:
            await asyncio.to_thread(thread_file.unlink)
        except Exception as e:
            raise StorageError(
                f"Failed to delete chat thread {thread_id}: {e}",
                details={"thread_id": thread_id},
            ) from e

    async def _write_thread(self, thread_file: Path, thread: StorageChatThread) -> None:
        """Write chat thread to JSON file atomically."""
        data = thread.model_dump(mode="json")

        def write_atomic():
            self._atomic_write_json(thread_file.parent, thread_file, data, indent=2)

        await asyncio.to_thread(write_atomic)

    @staticmethod
    def _atomic_write_json(
        parent_dir: Path,
        target: Path,
        data: object,
        indent: int | None = None,
    ) -> None:
        """Write JSON data to *target* atomically via temp-file + rename."""
        fd, tmp_path = tempfile.mkstemp(
            dir=str(parent_dir), suffix=".tmp", prefix=f".{target.stem}-"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(target))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    async def _read_thread(self, thread_file: Path) -> StorageChatThread:
        """Read chat thread from JSON file."""

        def read():
            with open(thread_file, encoding="utf-8") as f:
                return json.load(f)

        data = await asyncio.to_thread(read)
        return StorageChatThread.model_validate(data)
