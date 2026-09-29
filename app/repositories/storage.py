import logging
import os
from pathlib import Path
from typing import Optional

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class StorageRepository:
    """Repository handling PDF file persistence in Cloud Storage with local fallback."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._gcs_client = None
        self._bucket = None

        if self.settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                self.settings.google_application_credentials
            )

    @property
    def gcs_client(self):
        """Lazy-initialize GCS client if bucket is configured."""
        if self._gcs_client is None and self.settings.gcs_bucket_name:
            try:
                from google.cloud import storage
                project = self.settings.gcp_project_id or None
                self._gcs_client = storage.Client(project=project)
                self._bucket = self._gcs_client.bucket(self.settings.gcs_bucket_name)
                logger.info(f"Initialized Cloud Storage client with bucket: {self.settings.gcs_bucket_name}")
            except Exception as e:
                logger.warning(f"Could not initialize Cloud Storage client: {e}. Falling back to local disk storage.")
                self._gcs_client = None
        return self._gcs_client

    def build_storage_path(self, document_id: str, version_id: str, filename: str = "original.pdf") -> str:
        """Standard storage path: documents/{document_id}/{version_id}/{filename}."""
        return f"documents/{document_id}/{version_id}/{filename}"

    async def save_file(
        self,
        document_id: str,
        version_id: str,
        content: bytes,
        filename: str = "original.pdf",
    ) -> str:
        """Save file bytes to Cloud Storage or local fallback directory."""
        relative_path = self.build_storage_path(document_id, version_id, filename)

        if self.gcs_client and self._bucket:
            blob = self._bucket.blob(relative_path)
            blob.upload_from_string(content, content_type="application/pdf")
            gcs_uri = f"gs://{self.settings.gcs_bucket_name}/{relative_path}"
            logger.info(f"Stored file in GCS: {gcs_uri}")
            return gcs_uri

        # Local storage fallback
        local_dir = Path(self.settings.local_storage_dir) / "documents" / document_id / version_id
        local_dir.mkdir(parents=True, exist_ok=True)
        file_path = local_dir / filename
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"Stored file on local filesystem: {file_path}")
        return str(file_path.resolve())

    async def get_file(self, storage_path: str) -> bytes:
        """Retrieve file bytes from Cloud Storage or local filesystem."""
        if storage_path.startswith("gs://"):
            parts = storage_path[5:].split("/", 1)
            bucket_name = parts[0]
            blob_path = parts[1]
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            return blob.download_as_bytes()

        # Local path
        path = Path(storage_path)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found: {storage_path}")
        with open(path, "rb") as f:
            return f.read()


_storage_repo_instance: Optional[StorageRepository] = None


def get_storage_repository() -> StorageRepository:
    """Dependency provider for StorageRepository."""
    global _storage_repo_instance
    if _storage_repo_instance is None:
        _storage_repo_instance = StorageRepository()
    return _storage_repo_instance
