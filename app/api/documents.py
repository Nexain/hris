import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from app.core.errors import (
    DuplicateDocumentException,
    InvalidStateTransitionException,
    ResourceNotFoundException,
)
from app.core.security import UserContext, get_current_user, require_hr_admin
from app.models.document import (
    AuditLog,
    Document,
    DocumentChunk,
    DocumentCreate,
    DocumentProcessResponse,
    DocumentType,
    DocumentVersion,
    DocumentVersionUploadResponse,
    VersionStatus,
)
from app.repositories.firestore import FirestoreRepository, get_firestore_repository
from app.services.document_processor import (
    DocumentProcessor,
    get_document_processor,
)
from app.services.evaluation import RetrievalEvaluator
from app.services.retrieval import RetrievalService, get_retrieval_service

router = APIRouter(tags=["Documents"])


def slugify(text: str) -> str:
    """Generate a clean slug identifier from document title."""
    slug = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", slug)


@router.post(
    "/documents",
    response_model=Document,
    status_code=status.HTTP_201_CREATED,
    summary="Create a logical document",
    description="Creates a logical document entry in the company knowledge base.",
)
async def create_document(
    doc_in: DocumentCreate,
    current_user: UserContext = Depends(require_hr_admin),
    repo: FirestoreRepository = Depends(get_firestore_repository),
):
    """Create a new logical document."""
    doc_id = slugify(doc_in.title)
    existing = await repo.get_document(doc_id)
    if existing:
        # If slug exists, append a random short suffix
        doc_id = f"{doc_id}-{uuid.uuid4().hex[:4]}"

    doc = Document(
        id=doc_id,
        title=doc_in.title,
        document_type=doc_in.document_type,
        department=doc_in.department,
        location=doc_in.location,
        owner=doc_in.owner or current_user.name,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    created = await repo.create_document(doc)

    await repo.log_audit(
        AuditLog(
            id=str(uuid.uuid4()),
            actor_id=current_user.user_id,
            action="CREATE_DOCUMENT",
            resource_type="document",
            resource_id=doc.id,
            metadata={"title": doc.title, "department": doc.department},
        )
    )
    return created


@router.get(
    "/documents",
    response_model=List[Document],
    summary="List documents",
    description="List all logical documents with optional filtering by department, location, document type, and version status.",
)
async def list_documents(
    department: Optional[str] = Query(None, description="Filter by department"),
    location: Optional[str] = Query(None, description="Filter by location"),
    document_type: Optional[DocumentType] = Query(None, description="Filter by document type"),
    status: Optional[str] = Query(None, description="Filter by version status (e.g. PUBLISHED, REVIEW)"),
    repo: FirestoreRepository = Depends(get_firestore_repository),
):
    """Retrieve filtered list of documents."""
    doc_type_val = document_type.value if document_type else None
    return await repo.list_documents(
        department=department,
        location=location,
        document_type=doc_type_val,
        status=status,
    )


@router.get(
    "/documents/{document_id}",
    response_model=Document,
    summary="Get document details",
    description="Retrieve a single document by ID including all its recorded versions.",
)
async def get_document(
    document_id: str,
    repo: FirestoreRepository = Depends(get_firestore_repository),
):
    """Fetch logical document by ID."""
    doc = await repo.get_document(document_id)
    if not doc:
        raise ResourceNotFoundException("Document", document_id)
    return doc


@router.post(
    "/documents/{document_id}/versions",
    response_model=DocumentVersionUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a new document version PDF",
    description="Upload a PDF file as a new document version. Computes file SHA-256 and content SHA-256 for duplicate detection, stores file in storage, and initializes processing record.",
)
async def upload_document_version(
    document_id: str,
    file: UploadFile = File(..., description="PDF document file"),
    version: Optional[str] = Form(None, description="Semantic version string, e.g. '1.0'"),
    current_user: UserContext = Depends(require_hr_admin),
    processor: DocumentProcessor = Depends(get_document_processor),
):
    """Upload and validate a new PDF document version."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents (.pdf) are supported.",
        )

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    version_record = await processor.upload_version(
        document_id=document_id,
        pdf_bytes=pdf_bytes,
        version_str=version,
        uploaded_by=current_user.user_id,
    )

    return DocumentVersionUploadResponse(
        document_id=version_record.document_id,
        version_id=version_record.id,
        version=version_record.version,
        status=version_record.status,
        file_hash=version_record.file_hash,
        content_hash=version_record.content_hash,
        storage_path=version_record.storage_path,
        message="Version uploaded successfully and ready for processing.",
    )


@router.post(
    "/versions/{version_id}/process",
    response_model=DocumentProcessResponse,
    summary="Process document version into chunks",
    description="Extracts text, normalizes, detects headings/sections, chunks content, generates embeddings, and saves chunks to Firestore.",
)
async def process_version(
    version_id: str,
    current_user: UserContext = Depends(require_hr_admin),
    processor: DocumentProcessor = Depends(get_document_processor),
):
    """Run full ingestion and chunking pipeline on uploaded version."""
    return await processor.process_version(version_id)


@router.get(
    "/versions/{version_id}/chunks",
    response_model=List[DocumentChunk],
    summary="Get chunks for a document version",
    description="Retrieve all stored searchable chunks for a specific document version.",
)
async def get_version_chunks(
    version_id: str,
    repo: FirestoreRepository = Depends(get_firestore_repository),
):
    """Retrieve chunks belonging to a version."""
    version = await repo.get_document_version(version_id)
    if not version:
        raise ResourceNotFoundException("DocumentVersion", version_id)
    return await repo.get_chunks_by_version(version_id)


@router.get(
    "/versions/{version_id}/evaluation",
    summary="Evaluate a document version against persisted golden questions",
    description="Runs the document's persisted evaluation questions against this version's chunks and returns Hit@K / MRR (spec §16.4 regression testing).",
)
async def evaluate_version(
    version_id: str,
    top_k: int = Query(5, ge=1, le=20, description="Top-K for retrieval evaluation"),
    repo: FirestoreRepository = Depends(get_firestore_repository),
    retrieval: RetrievalService = Depends(get_retrieval_service),
):
    """Run regression evaluation for a version's chunks."""
    version = await repo.get_document_version(version_id)
    if not version:
        raise ResourceNotFoundException("DocumentVersion", version_id)

    evaluator = RetrievalEvaluator(retrieval_service=retrieval)
    result = await evaluator.evaluate_version(version.document_id, version_id, repo, top_k=top_k)

    return {
        "document_id": version.document_id,
        "version_id": version_id,
        "total": result.total,
        "hits": result.hits,
        "hit_rate": result.hit_rate,
        "mrr": result.mrr,
        "top_k": result.top_k,
        "passed": result.passed,
        "results": [
            {
                "question": r.question,
                "expected_document_id": r.expected_document_id,
                "retrieved_document_ids": r.retrieved_document_ids,
                "hit": r.hit,
                "rank": r.rank,
            }
            for r in result.results
        ],
    }


@router.post(
    "/versions/{version_id}/publish",
    response_model=DocumentVersion,
    summary="Publish document version",
    description="Marks a reviewed document version as PUBLISHED and supersedes any previously published version.",
)
async def publish_version(
    version_id: str,
    current_user: UserContext = Depends(require_hr_admin),
    repo: FirestoreRepository = Depends(get_firestore_repository),
):
    """Publish a document version."""
    target_version = await repo.get_document_version(version_id)
    if not target_version:
        raise ResourceNotFoundException("DocumentVersion", version_id)

    # Spec §4.4: only a version that passed processing/validation (REVIEW) may be
    # published. Upload does not equal publish.
    if target_version.status != VersionStatus.REVIEW:
        raise InvalidStateTransitionException(
            f"Only versions in REVIEW may be published; version '{version_id}' is "
            f"'{target_version.status.value}'."
        )

    # Fetch document
    doc = await repo.get_document(target_version.document_id)
    if not doc:
        raise ResourceNotFoundException("Document", target_version.document_id)

    # Supersede previously published version if any
    all_versions = await repo.get_document_versions(doc.id)
    for v in all_versions:
        if v.id != version_id and v.status == VersionStatus.PUBLISHED:
            v.status = VersionStatus.SUPERSEDED
            await repo.update_document_version(v)

    target_version.status = VersionStatus.PUBLISHED
    target_version.published_at = datetime.now(timezone.utc)
    await repo.update_document_version(target_version)

    doc.current_version_id = target_version.id
    doc.updated_at = datetime.now(timezone.utc)
    await repo.create_document(doc)

    await repo.log_audit(
        AuditLog(
            id=str(uuid.uuid4()),
            actor_id=current_user.user_id,
            action="PUBLISH_VERSION",
            resource_type="document_version",
            resource_id=version_id,
            new_state=VersionStatus.PUBLISHED.value,
        )
    )

    return target_version


@router.post(
    "/versions/{version_id}/reject",
    response_model=DocumentVersion,
    summary="Reject document version",
    description="Marks a document version as REJECTED so it cannot be published or retrieved.",
)
async def reject_version(
    version_id: str,
    current_user: UserContext = Depends(require_hr_admin),
    repo: FirestoreRepository = Depends(get_firestore_repository),
):
    """Reject a document version."""
    target_version = await repo.get_document_version(version_id)
    if not target_version:
        raise ResourceNotFoundException("DocumentVersion", version_id)

    # Spec §9: a published/superseded version cannot be rejected.
    if target_version.status in (VersionStatus.PUBLISHED, VersionStatus.SUPERSEDED):
        raise InvalidStateTransitionException(
            f"Cannot reject a '{target_version.status.value}' version."
        )

    previous_state = target_version.status.value
    target_version.status = VersionStatus.REJECTED
    await repo.update_document_version(target_version)

    await repo.log_audit(
        AuditLog(
            id=str(uuid.uuid4()),
            actor_id=current_user.user_id,
            action="REJECT_VERSION",
            resource_type="document_version",
            resource_id=version_id,
            previous_state=previous_state,
            new_state=VersionStatus.REJECTED.value,
        )
    )

    return target_version
