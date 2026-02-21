"""Tests for FileStorageClient."""

from unittest.mock import MagicMock, patch

import pytest

from harkd.config import FileStorageSettings
from harkd.transcription.file_storage import FileStorageClient


@pytest.fixture
def storage_settings():
    """Create test file storage settings."""
    return FileStorageSettings(
        endpoint="https://s3.example.com",
        bucket="test-bucket",
        access_key="access-key",
        secret_key="secret-key",
        region="eu-west-1",
    )


class TestFileStorageClient:
    """Tests for S3-compatible file storage."""

    @patch("harkd.transcription.file_storage.boto3")
    def test_init_creates_s3_client(self, mock_boto3, storage_settings):
        """Test S3 client is created with correct settings."""
        FileStorageClient(storage_settings)

        mock_boto3.client.assert_called_once_with(
            "s3",
            endpoint_url="https://s3.example.com",
            aws_access_key_id="access-key",
            aws_secret_access_key="secret-key",
            region_name="eu-west-1",
        )

    @pytest.mark.asyncio
    @patch("harkd.transcription.file_storage.boto3")
    async def test_upload_returns_url_and_key(self, mock_boto3, storage_settings, tmp_path):
        """Test upload returns presigned URL and object key."""
        mock_s3 = MagicMock()
        mock_s3.generate_presigned_url.return_value = "https://s3.example.com/presigned"
        mock_boto3.client.return_value = mock_s3

        client = FileStorageClient(storage_settings)
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"audio data")

        url, key = await client.upload(audio_file)

        assert url == "https://s3.example.com/presigned"
        assert key.startswith("harkd-audio/")
        assert key.endswith("/test.wav")
        mock_s3.upload_file.assert_called_once()
        mock_s3.generate_presigned_url.assert_called_once()

    @pytest.mark.asyncio
    @patch("harkd.transcription.file_storage.boto3")
    async def test_delete_calls_s3(self, mock_boto3, storage_settings):
        """Test delete calls S3 delete_object."""
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        client = FileStorageClient(storage_settings)
        await client.delete("harkd-audio/uuid/test.wav")

        mock_s3.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="harkd-audio/uuid/test.wav"
        )
