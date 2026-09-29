import os
from functools import lru_cache
from typing import Optional
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="DayOne AI Backend", validation_alias=AliasChoices("APP_NAME", "app_name"))
    app_env: str = Field(default="development", validation_alias=AliasChoices("APP_ENV", "app_env", "ENVIRONMENT"))
    port: int = Field(default=8000, validation_alias=AliasChoices("PORT", "port"))
    host: str = Field(default="0.0.0.0", validation_alias=AliasChoices("HOST", "host"))
    log_level: str = Field(default="INFO", validation_alias=AliasChoices("LOG_LEVEL", "log_level"))

    # Google Cloud Project & Location
    gcp_project_id: str = Field(
        default="",
        validation_alias=AliasChoices("GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "gcp_project_id"),
    )
    gcp_location: str = Field(
        default="us-central1",
        validation_alias=AliasChoices("GCP_LOCATION", "GOOGLE_CLOUD_LOCATION", "gcp_location"),
    )

    # Models
    gemini_model: str = Field(default="gemini-1.5-flash", validation_alias=AliasChoices("GEMINI_MODEL", "gemini_model"))
    embedding_model: str = Field(
        default="text-embedding-004",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "embedding_model"),
    )

    # Cloud Storage & Firestore
    gcs_bucket_name: str = Field(default="", validation_alias=AliasChoices("GCS_BUCKET_NAME", "gcs_bucket_name"))
    firestore_database: str = Field(
        default="(default)",
        validation_alias=AliasChoices("FIRESTORE_DATABASE", "firestore_database"),
    )

    # Optional Service Account Credentials
    google_application_credentials: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_APPLICATION_CREDENTIALS", "google_application_credentials"),
    )

    # Local fallback directory for documents when GCS is not configured
    local_storage_dir: str = Field(
        default="./data/storage",
        validation_alias=AliasChoices("LOCAL_STORAGE_DIR", "local_storage_dir"),
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
