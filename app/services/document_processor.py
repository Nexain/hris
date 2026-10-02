import hashlib
import io
import logging
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pypdf import PdfReader

from app.core.config import Settings, get_settings
from app.core.errors import (
    DocumentProcessingException,
    DuplicateDocumentException,
    ResourceNotFoundException,
)
from app.models.document import (
    AuditLog,
    Document,
    DocumentChunk,
    DocumentProcessResponse,
    DocumentVersion,
    EvaluationQuestion,
    VersionStatus,
)
from app.repositories.firestore import FirestoreRepository, get_firestore_repository
from app.repositories.storage import StorageRepository, get_storage_repository
from app.services.embeddings import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Service handling PDF ingestion: hashing, duplicate detection, extraction, normalization, chunking, and storage."""

    def __init__(
        self,
        firestore_repo: Optional[FirestoreRepository] = None,
        storage_repo: Optional[StorageRepository] = None,
        embedding_service: Optional[EmbeddingService] = None,
        settings: Optional[Settings] = None,
    ):
        self.firestore_repo = firestore_repo or get_firestore_repository()
        self.storage_repo = storage_repo or get_storage_repository()
        self.embedding_service = embedding_service or get_embedding_service()
        self.settings = settings or get_settings()

    # ------------------ Utility / Extraction / Normalization ------------------
    @staticmethod
    def calculate_bytes_hash(data: bytes) -> str:
        """Calculate SHA-256 of raw bytes."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def calculate_text_hash(text: str) -> str:
        """Calculate SHA-256 of normalized text string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text: Unicode NFKC, clean whitespace, uniform newlines."""
        if not text:
            return ""
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        # Uniform newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Replace multiple horizontal spaces/tabs with single space
        text = re.sub(r"[ \t]+", " ", text)
        # Strip trailing/leading spaces on lines
        lines = [line.strip() for line in text.split("\n")]
        # Compress multiple blank lines to at most two
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
        return text.strip()

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract text page by page from PDF bytes."""
        try:
            stream = io.BytesIO(pdf_bytes)
            reader = PdfReader(stream)
            pages: List[Dict[str, Any]] = []

            for idx, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                norm_text = self.normalize_text(raw_text)
                if norm_text:
                    pages.append({"page": idx + 1, "text": norm_text})

            if not pages:
                raise DocumentProcessingException(
                    "No extractable text found in the provided PDF file."
                )

            return pages
        except DocumentProcessingException:
            raise
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise DocumentProcessingException(f"Failed to parse PDF document: {str(e)}")

    # ------------------ Structure & Chunking ------------------
    @staticmethod
    def is_likely_heading(line: str) -> bool:
        """Heuristic to detect headings and section titles."""
        line = line.strip()
        if not line or len(line) > 100:
            return False

        # Markdown headings
        if line.startswith("#"):
            return True

        # Numbered headings e.g. "1. Introduction", "1.1 General Policy", "Section 3"
        if re.match(r"^(\d+(\.\d+)*\.?|[A-Z]\.|\bSection\b|\bArticle\b|\bChapter\b|\bPart\b)\s*[:.-]?\s*", line, re.IGNORECASE):
            return True

        # ALL CAPS short line (at least 4 letters)
        letters = [c for c in line if c.isalpha()]
        if len(letters) >= 4 and line.isupper() and len(line) < 60:
            return True

        # Title Case short line without ending punctuation
        if (
            len(line.split()) <= 8
            and not line.endswith((".", ",", ";", ":", "-"))
            and line[0].isupper()
            and all(w[0].isupper() for w in line.split() if len(w) > 3)
        ):
            return True

        return False

    def chunk_document(
        self,
        pages: List[Dict[str, Any]],
        target_chunk_chars: int = 3000,  # ~800 tokens
        overlap_chars: int = 400,        # ~100 tokens
    ) -> List[Dict[str, Any]]:
        """Heading-aware chunking preserving section titles and page numbers."""
        # 1. Segment text into structured blocks with current heading
        blocks: List[Dict[str, Any]] = []
        current_section = "General"

        for p in pages:
            page_num = p["page"]
            paragraphs = p["text"].split("\n\n")

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # Check if paragraph is heading
                lines = para.split("\n")
                if len(lines) == 1 and self.is_likely_heading(lines[0]):
                    current_section = re.sub(r"^#+\s*", "", lines[0]).strip()
                    continue

                # Check if first line of paragraph is heading
                if len(lines) > 1 and self.is_likely_heading(lines[0]):
                    current_section = re.sub(r"^#+\s*", "", lines[0]).strip()
                    para = "\n".join(lines[1:]).strip()
                    if not para:
                        continue

                blocks.append({
                    "section": current_section,
                    "page": page_num,
                    "text": para,
                })

        if not blocks:
            return []

        # 2. Assemble blocks into chunks up to target_chunk_chars
        chunks: List[Dict[str, Any]] = []
        current_chunk_text = ""
        current_chunk_section = blocks[0]["section"]
        current_chunk_page = blocks[0]["page"]

        for block in blocks:
            block_text = block["text"]
            block_section = block["section"]
            block_page = block["page"]

            # Break on new section if current chunk already has text, or when exceeding target size
            section_changed = (block_section != current_chunk_section and bool(current_chunk_text))
            size_exceeded = (len(current_chunk_text) + len(block_text) + 2 > target_chunk_chars and bool(current_chunk_text))

            if section_changed or size_exceeded:
                chunks.append({
                    "content": current_chunk_text.strip(),
                    "section": current_chunk_section,
                    "page": current_chunk_page,
                })

                if not section_changed and len(current_chunk_text) > overlap_chars:
                    # Calculate overlap within same section
                    overlap_seed = current_chunk_text[-overlap_chars:]
                    space_idx = overlap_seed.find(" ")
                    if space_idx != -1:
                        overlap_seed = overlap_seed[space_idx + 1:]
                    current_chunk_text = overlap_seed + "\n\n" + block_text
                else:
                    current_chunk_text = block_text

                current_chunk_section = block_section
                current_chunk_page = block_page
            else:
                if current_chunk_text:
                    current_chunk_text += "\n\n" + block_text
                else:
                    current_chunk_text = block_text
                    current_chunk_section = block_section
                    current_chunk_page = block_page

        if current_chunk_text.strip():
            chunks.append({
                "content": current_chunk_text.strip(),
                "section": current_chunk_section,
                "page": current_chunk_page,
            })

        return chunks

    # ------------------ Evaluation Question Generation (spec §6.5, §4.3) ------------------
    @staticmethod
    def build_evaluation_questions(
        document: Document,
        chunks: List[DocumentChunk],
        max_questions: int = 5,
    ) -> List[EvaluationQuestion]:
        """Derive regression questions from a document's distinct sections.

        Questions are tied to the document (not a version) so they persist and can
        be re-run against future versions (spec §16.4).
        """
        questions: List[EvaluationQuestion] = []
        seen_sections = set()

        for chunk in chunks:
            section = (chunk.section or "").strip()
            key = section.lower()
            if not section or key in seen_sections:
                continue
            seen_sections.add(key)

            questions.append(
                EvaluationQuestion(
                    id=f"{document.id}_eq{len(questions):03d}",
                    document_id=document.id,
                    question=f"What does the document say about {section}?",
                    expected_source=document.title,
                )
            )
            if len(questions) >= max_questions:
                break

        return questions

    async def generate_evaluation_questions(
        self,
        document: Document,
        chunks: List[DocumentChunk],
    ) -> List[EvaluationQuestion]:
        """Generate and persist evaluation questions once per document.

        Existing questions are reused so regressions can be tracked across versions
        without overwriting the golden set (spec §16.4).
        """
        existing = await self.firestore_repo.get_evaluation_questions(document.id)
        if existing:
            return existing

        questions = self.build_evaluation_questions(document, chunks)
        if questions:
            await self.firestore_repo.save_evaluation_questions(questions)
        return questions

    # ------------------ Upload & Ingestion Workflows ------------------
    async def upload_version(
        self,
        document_id: str,
        pdf_bytes: bytes,
        version_str: Optional[str] = None,
        uploaded_by: str = "hr_admin",
    ) -> DocumentVersion:
        """Handle document version upload with duplicate checks."""
        # Verify logical document exists
        document = await self.firestore_repo.get_document(document_id)
        if not document:
            raise ResourceNotFoundException("Document", document_id)

        # 1. Calculate file SHA-256
        file_hash = self.calculate_bytes_hash(pdf_bytes)

        # 2. Check for duplicate file hash
        existing_file = await self.firestore_repo.find_version_by_file_hash(file_hash)
        if existing_file:
            raise DuplicateDocumentException(
                f"Duplicate file detected: Exactly identical PDF already uploaded in version '{existing_file.id}' of document '{existing_file.document_id}'."
            )

        # Determine version string
        existing_versions = await self.firestore_repo.get_document_versions(document_id)
        if not version_str:
            version_str = f"{len(existing_versions) + 1}.0"

        version_id = f"v{version_str.replace('.', '_')}_{uuid.uuid4().hex[:6]}"

        # 3. Store PDF in Cloud Storage
        storage_path = await self.storage_repo.save_file(
            document_id=document_id,
            version_id=version_id,
            content=pdf_bytes,
        )

        # 4. Extract text
        pages = self.extract_text_from_pdf(pdf_bytes)

        # 5. Normalize text & 6. Calculate content hash
        full_normalized = "\n\n".join(p["text"] for p in pages)
        content_hash = self.calculate_text_hash(full_normalized)

        # 7. Check for duplicate content hash
        existing_content = await self.firestore_repo.find_version_by_content_hash(content_hash)
        if existing_content:
            raise DuplicateDocumentException(
                f"Duplicate content detected: Identical extracted text content already exists in version '{existing_content.id}' of document '{existing_content.document_id}'."
            )

        # 8. Create version record
        version = DocumentVersion(
            id=version_id,
            document_id=document_id,
            version=version_str,
            status=VersionStatus.PROCESSING,
            file_hash=file_hash,
            content_hash=content_hash,
            storage_path=storage_path,
            uploaded_by=uploaded_by,
            created_at=datetime.now(timezone.utc),
        )
        await self.firestore_repo.create_document_version(version)

        # Audit log
        await self.firestore_repo.log_audit(
            AuditLog(
                id=str(uuid.uuid4()),
                actor_id=uploaded_by,
                action="UPLOAD_VERSION",
                resource_type="document_version",
                resource_id=version_id,
                new_state=VersionStatus.PROCESSING.value,
                metadata={"document_id": document_id, "file_hash": file_hash, "version": version_str},
            )
        )

        return version

    async def process_version(self, version_id: str) -> DocumentProcessResponse:
        """Run full processing pipeline on an uploaded version: chunking, embeddings, Firestore persistence."""
        version = await self.firestore_repo.get_document_version(version_id)
        if not version:
            raise ResourceNotFoundException("DocumentVersion", version_id)

        document = await self.firestore_repo.get_document(version.document_id)
        if not document:
            raise ResourceNotFoundException("Document", version.document_id)

        try:
            # 1. Fetch PDF from storage
            pdf_bytes = await self.storage_repo.get_file(version.storage_path)

            # 2. Extract text & normalize
            pages = self.extract_text_from_pdf(pdf_bytes)

            # 3. Heading-aware chunking
            raw_chunks = self.chunk_document(pages)

            if not raw_chunks:
                raise DocumentProcessingException("Chunking produced 0 chunks for document.")

            # 4. Generate embeddings for all chunks
            chunk_texts = [c["content"] for c in raw_chunks]
            embeddings = await self.embedding_service.get_embeddings(chunk_texts)

            # 5. Build DocumentChunk entities
            chunks: List[DocumentChunk] = []
            for idx, c in enumerate(raw_chunks):
                chunk_id = f"{version_id}_c{idx:03d}"
                metadata = {
                    "document_title": document.title,
                    "document_type": document.document_type.value,
                    "department": document.department,
                    "location": document.location,
                    "version": version.version,
                }
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        document_id=version.document_id,
                        version_id=version.id,
                        content=c["content"],
                        section=c["section"],
                        page=c["page"],
                        chunk_index=idx,
                        embedding=embeddings[idx] if idx < len(embeddings) else None,
                        metadata=metadata,
                        created_at=datetime.now(timezone.utc),
                    )
                )

            # 6. Delete old chunks if reprocessing and save new chunks
            await self.firestore_repo.delete_chunks_by_version(version_id)
            await self.firestore_repo.save_chunks(chunks)

            # 7. Generate/persist evaluation questions (once per document)
            eval_questions = await self.generate_evaluation_questions(document, chunks)

            # 8. Update version status to REVIEW (automated validation passed, ready for HR review/publishing)
            version.status = VersionStatus.REVIEW
            version.chunk_count = len(chunks)
            version.processed_at = datetime.now(timezone.utc)
            await self.firestore_repo.update_document_version(version)

            # Audit log
            await self.firestore_repo.log_audit(
                AuditLog(
                    id=str(uuid.uuid4()),
                    actor_id=version.uploaded_by,
                    action="PROCESS_VERSION",
                    resource_type="document_version",
                    resource_id=version_id,
                    previous_state=VersionStatus.PROCESSING.value,
                    new_state=VersionStatus.REVIEW.value,
                    metadata={"chunk_count": len(chunks), "evaluation_questions": len(eval_questions)},
                )
            )

            return DocumentProcessResponse(
                document_id=version.document_id,
                version_id=version.id,
                status=VersionStatus.REVIEW,
                chunk_count=len(chunks),
                chunks=chunks,
                message="Document processed into searchable chunks successfully and ready for HR review.",
            )

        except Exception as e:
            logger.error(f"Processing version '{version_id}' failed: {e}")
            version.status = VersionStatus.FAILED
            version.error_message = str(e)
            await self.firestore_repo.update_document_version(version)
            raise DocumentProcessingException(f"Processing version failed: {str(e)}")


_document_processor_instance: Optional[DocumentProcessor] = None


def get_document_processor() -> DocumentProcessor:
    """Dependency provider for DocumentProcessor."""
    global _document_processor_instance
    if _document_processor_instance is None:
        _document_processor_instance = DocumentProcessor()
    return _document_processor_instance
