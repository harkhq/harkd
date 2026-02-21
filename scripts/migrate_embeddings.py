#!/usr/bin/env python3
"""Migrate speaker_embeddings from metadata.json to separate embeddings.json files.

Standalone script (no harkd imports). Safe to run multiple times (idempotent).

Usage:
    uv run python scripts/migrate_embeddings.py --dry-run
    uv run python scripts/migrate_embeddings.py
    uv run python scripts/migrate_embeddings.py --base-path /custom/path
"""

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path


def atomic_write_json(
    parent_dir: Path, target: Path, data: object, indent: int | None = None
) -> None:
    """Write JSON to *target* atomically via temp-file + os.replace."""
    fd, tmp_path = tempfile.mkstemp(dir=str(parent_dir), suffix=".tmp", prefix=f".{target.stem}-")
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


def migrate_recording(rec_dir: Path, *, dry_run: bool) -> bool:
    """Migrate a single recording directory. Returns True if migration was performed."""
    metadata_file = rec_dir / "metadata.json"
    embeddings_file = rec_dir / "embeddings.json"

    if not metadata_file.exists():
        return False

    with open(metadata_file, encoding="utf-8") as f:
        data = json.load(f)

    embeddings = data.get("speaker_embeddings")
    if embeddings is None:
        return False

    if dry_run:
        print(f"  [dry-run] Would migrate: {rec_dir.name} ({len(embeddings)} speakers)")
        return True

    # 1. Write embeddings.json first (crash-safe ordering)
    atomic_write_json(rec_dir, embeddings_file, embeddings)

    # 2. Rewrite metadata.json without speaker_embeddings
    del data["speaker_embeddings"]
    atomic_write_json(rec_dir, metadata_file, data, indent=2)

    print(f"  Migrated: {rec_dir.name} ({len(embeddings)} speakers)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate speaker_embeddings to separate files")
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path.home() / ".local" / "share" / "hark",
        help="Base storage path (default: ~/.local/share/hark)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )
    args = parser.parse_args()

    recordings_dir = args.base_path / "recordings"
    if not recordings_dir.exists():
        print(f"No recordings directory found at {recordings_dir}")
        sys.exit(0)

    print(f"Scanning {recordings_dir} ...")
    if args.dry_run:
        print("(dry-run mode — no files will be changed)\n")

    migrated = 0
    total = 0

    for entry in sorted(recordings_dir.iterdir()):
        if not entry.is_dir():
            continue
        total += 1
        if migrate_recording(entry, dry_run=args.dry_run):
            migrated += 1

    print(f"\nDone. {migrated}/{total} recordings migrated.")


if __name__ == "__main__":
    main()
