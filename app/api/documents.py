"""
Documents API — Company Knowledge Base Management
==================================================

This module handles the full lifecycle of HR documents (policies, guides, SOPs)
that feed the RAG (Retrieval-Augmented Generation) pipeline powering the AI chat.

Document Lifecycle (must follow this order):
--------------------------------------------
  1. POST /documents                  → Create a logical document record (HR Admin only)
  2. POST /documents/{id}/versions    → Upload a PDF file for that document
  3. POST /versions/{id}/process      → Extract text, chunk it, and generate embeddings
  4. GET  /versions/{id}/chunks       → (Optional) Inspect the chunks that were produced
  5. GET  /versions/{id}/evaluation   → (Optional) Run quality check on retrieval accuracy
  6. POST /versions/{id}/publish      → Make the version live so the chat AI can use it
  7. POST /versions/{id}/reject       → Reject a version that failed review

Key Concepts:
-------------
- A "Document" is a logical entry (e.g. "Leave Policy"). It can have many versions.
- A "Version" is a specific PDF upload. Only ONE version can be PUBLISHED at a time.
  When you publish a new version, the previous one becomes SUPERSEDED automatically.
- The chat AI (RAG) only searches chunks from PUBLISHED versions. Drafts are invisible.
- All write operations require HR_ADMIN role (pass header: X-Access-Level: HR_ADMIN).
"""

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
    """
    STEP 1 of the document lifecycle — HR Admin creates a logical document entry.

    This does NOT upload any file yet. It just registers the document's metadata
    (title, type, department, location) in Firestore and assigns it a slug-based ID
    (e.g. "Leave Policy" → "leave-policy").

    After this, the HR Admin should upload a PDF using POST /documents/{id}/versions.
    Every write action is recorded in the audit log for traceability.

    Access: HR_ADMIN only.
    """
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
    """
    Returns a filtered list of all logical documents in the knowledge base.

    Each document in the response also includes its version history, so you can
    see which versions are PUBLISHED, in REVIEW, SUPERSEDED, or REJECTED.

    Useful for HR Admins to see what content is currently in the knowledge base,
    and for debugging which documents are available to the chat AI (PUBLISHED only).

    Access: Public (no auth required — read-only listing).
    """
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
    """
    Returns the full details of a single document by its slug ID (e.g. "leave-policy").

    The response includes all versions of the document and their statuses.
    Use this to check the current state of a document before deciding to upload
    a new version or publish an existing one.

    Access: Public (no auth required — read-only).
    """
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
    """
    STEP 2 of the document lifecycle — Upload a PDF file as a new version of a document.

    What this does:
    - Validates the file is a non-empty PDF
    - Computes a SHA-256 hash of the file bytes to detect exact duplicates
    - Computes a SHA-256 hash of the extracted text to detect content duplicates (same content, different filename)
    - Uploads the raw PDF to Google Cloud Storage
    - Creates a DocumentVersion record in Firestore with status UPLOADED

    After this step, the version is stored but NOT searchable yet.
    The HR Admin must call POST /versions/{id}/process next to extract and embed the text.

    Access: HR_ADMIN only.
    """
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
    """
    STEP 3 of the document lifecycle — Run the full AI ingestion pipeline on a version.

    What this does (in order):
    1. Downloads the PDF from Cloud Storage
    2. Extracts and normalizes the raw text
    3. Splits the text into overlapping chunks (paragraphs/sections)
    4. Generates a vector embedding for each chunk using Vertex AI Embeddings
    5. Saves all chunks with their embeddings to Firestore
    6. Moves the version status from UPLOADED → REVIEW

    After this step, the version is ready to be reviewed and published.
    The version is still NOT searchable by the chat AI until it is published.

    Access: HR_ADMIN only.
    """
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
    """
    STEP 4 (Optional) — Inspect the text chunks generated from processing a version.

    After calling POST /versions/{id}/process, the PDF is broken into text chunks.
    This endpoint lets you read those chunks to verify the text was extracted correctly
    before publishing the version.

    Each chunk contains: the text content, its position index, section heading,
    and the vector embedding used for similarity search.

    Access: Public (read-only, useful for HR review and debugging).
    """
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
    """
    STEP 5 (Optional) — Run a quality check on how well this version answers known questions.

    Before publishing a new version, you can test whether the RAG retrieval actually
    finds the right chunks when asked pre-defined "golden" questions. This helps catch
    cases where text extraction went wrong or the content is poorly structured.

    Metrics returned:
    - hit_rate: % of questions where the correct chunk appeared in the top-K results
    - mrr: Mean Reciprocal Rank (how high the correct chunk ranked on average)
    - passed: True if hit_rate and mrr meet the minimum quality thresholds

    Use this to make a data-driven decision before calling publish.

    Access: Public (read-only evaluation, no data is modified).
    """
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
    """
    STEP 6 of the document lifecycle — Make a processed version live and searchable.

    This is the final step. Once published:
    - The version status changes to PUBLISHED
    - Any previously PUBLISHED version of the same document becomes SUPERSEDED
    - The document's current_version_id pointer is updated
    - The chat AI (RAG) can now find and use this version's chunks to answer questions

    Precondition: the version must be in REVIEW status (i.e. it has been processed).
    You cannot publish a version that is still UPLOADED, already PUBLISHED, or REJECTED.

    Every publish action is written to the audit log.

    Access: HR_ADMIN only.
    """
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
    """
    STEP 6 (Alternative) — Reject a version instead of publishing it.

    Use this when the processed version has quality issues (e.g. text extraction was
    garbled, wrong file was uploaded, content failed the evaluation check).

    A REJECTED version:
    - Cannot be published or un-rejected
    - Is permanently excluded from RAG search
    - Remains in the version history for audit purposes

    Restrictions: you cannot reject a version that is already PUBLISHED or SUPERSEDED.
    To update a published document, upload a new version instead.

    Every rejection is written to the audit log.

    Access: HR_ADMIN only.
    """
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
