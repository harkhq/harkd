"""S3-compatible file storage for URL-based audio transfer."""

import asyncio
import logging
from pathlib import Path
from uuid import uuid4

import boto3

from harkd.config import FileStorageSettings

__all__ = ["FileStorageClient"]

logger = logging.getLogger(__name__)


class FileStorageClient:
    """S3-compatible file storage for URL-based audio transfer.

    Used by backends that need presigned URLs (e.g. Verda) instead of
    direct file upload.
    """

    def __init__(self, settings: FileStorageSettings):
        self._s3 = boto3.client(
            "s3",
            endpoint_url=settings.endpoint,
            aws_access_key_id=settings.access_key,
            aws_secret_access_key=settings.secret_key,
            region_name=settings.region,
        )
        self._bucket = settings.bucket

    async def upload(self, file_path: Path) -> tuple[str, str]:
        """Upload file and return (presigned_url, object_key).

        The presigned URL expires after 1 hour.
        """
        key = f"harkd-audio/{uuid4()}/{file_path.name}"
        await asyncio.to_thread(self._s3.upload_file, str(file_path), self._bucket, key)
        url = self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=3600,
        )
        logger.debug(f"Uploaded {file_path.name} to s3://{self._bucket}/{key}")
        return url, key

    async def delete(self, key: str) -> None:
        """Delete object after transcription completes."""
        await asyncio.to_thread(self._s3.delete_object, Bucket=self._bucket, Key=key)
        logger.debug(f"Deleted s3://{self._bucket}/{key}")
