from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    POLICY = "POLICY"
    HANDBOOK = "HANDBOOK"
    PROCEDURE = "PROCEDURE"
    GUIDE = "GUIDE"
    BENEFITS = "BENEFITS"
    IT = "IT"
    OTHER = "OTHER"


class VersionStatus(str, Enum):
    DRAFT = "DRAFT"
    PROCESSING = "PROCESSING"
    FAILED = "FAILED"
    REVIEW = "REVIEW"
    REJECTED = "REJECTED"
    PUBLISHED = "PUBLISHED"
    SUPERSEDED = "SUPERSEDED"


# Request / Response Schemas for Documents
class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Title of the logical document")
    document_type: DocumentType = Field(default=DocumentType.POLICY, description="Type/category of the document")
    department: str = Field(default="People Operations", description="Owning department")
    location: Optional[str] = Field(default="Jakarta", description="Applicable location or office")
    owner: Optional[str] = Field(default="People Operations", description="Owner name or team")


class DocumentVersion(BaseModel):
    id: str = Field(..., description="Unique version identifier (e.g. v1, v2, or UUID)")
    document_id: str
    version: str = Field(..., description="Semantic version string, e.g., '1.0'")
    status: VersionStatus = Field(default=VersionStatus.PROCESSING)
    file_hash: str = Field(..., description="SHA-256 of the original PDF binary")
    content_hash: Optional[str] = Field(default=None, description="SHA-256 of extracted normalized text")
    storage_path: str = Field(..., description="Cloud Storage / local path to original PDF")
    effective_date: Optional[str] = None
    uploaded_by: str = "hr_admin"
    chunk_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: Optional[datetime] = None
    published_at: Optional[datetime] = None


class Document(BaseModel):
    id: str = Field(..., description="Logical document identifier, e.g. 'leave-policy'")
    title: str
    document_type: DocumentType
    department: str
    location: Optional[str] = None
    owner: Optional[str] = None
    current_version_id: Optional[str] = None
    versions: List[DocumentVersion] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentChunk(BaseModel):
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str
    version_id: str
    content: str = Field(..., description="Text content of this chunk")
    section: Optional[str] = Field(default=None, description="Detected document section/heading")
    page: Optional[int] = Field(default=None, description="PDF page number (1-indexed)")
    chunk_index: int = Field(..., description="Zero-based index of chunk in document")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata such as title, department, etc.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentVersionUploadResponse(BaseModel):
    document_id: str
    version_id: str
    version: str
    status: VersionStatus
    file_hash: str
    content_hash: Optional[str] = None
    storage_path: str
    message: str


class DocumentProcessResponse(BaseModel):
    document_id: str
    version_id: str
    status: VersionStatus
    chunk_count: int
    chunks: List[DocumentChunk] = Field(default_factory=list)
    message: str


class AuditLog(BaseModel):
    id: str
    actor_id: str
    action: str
    resource_type: str
    resource_id: str
    previous_state: Optional[str] = None
    new_state: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationQuestion(BaseModel):
    """A persistent regression question for a document (spec §6.5)."""

    id: str = Field(..., description="Unique evaluation question identifier")
    document_id: str = Field(..., description="Document this question evaluates")
    question: str = Field(..., description="Golden retrieval/answer question")
    expected_source: str = Field(..., description="Expected source document title or id")
    expected_answer: Optional[str] = Field(default=None, description="Optional expected answer snippet")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
