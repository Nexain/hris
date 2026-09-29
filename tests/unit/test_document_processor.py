import pytest

from app.core.errors import DocumentProcessingException, DuplicateDocumentException
from app.models.document import Document, DocumentType, VersionStatus
from app.repositories.firestore import FirestoreRepository
from app.repositories.storage import StorageRepository
from app.services.document_processor import DocumentProcessor
from app.services.embeddings import EmbeddingService
from tests.conftest import make_test_pdf


@pytest.fixture
def isolated_processor(tmp_path):
    """Create an isolated DocumentProcessor with clean in-memory Firestore and temp storage."""
    from app.core.config import Settings
    settings = Settings(local_storage_dir=str(tmp_path / "storage"))
    firestore_repo = FirestoreRepository(settings=settings)
    storage_repo = StorageRepository(settings=settings)
    embedding_service = EmbeddingService(settings=settings)
    return DocumentProcessor(
        firestore_repo=firestore_repo,
        storage_repo=storage_repo,
        embedding_service=embedding_service,
        settings=settings,
    )


def test_calculate_hashes():
    data = b"Hello World"
    h1 = DocumentProcessor.calculate_bytes_hash(data)
    h2 = DocumentProcessor.calculate_text_hash("Hello World")
    assert h1 == h2
    assert len(h1) == 64


def test_normalize_text():
    raw = "  Hello   World!  \r\n\r\n\n\nLine   2  with   tabs\t\there.  "
    normalized = DocumentProcessor.normalize_text(raw)
    assert "Hello World!" in normalized
    assert "Line 2 with tabs here." in normalized
    assert "\r" not in normalized
    assert "\n\n\n" not in normalized


def test_is_likely_heading():
    assert DocumentProcessor.is_likely_heading("1. Introduction") is True
    assert DocumentProcessor.is_likely_heading("1.2 Leave Policy") is True
    assert DocumentProcessor.is_likely_heading("Section 4: Laptop Policy") is True
    assert DocumentProcessor.is_likely_heading("# Core Principles") is True
    assert DocumentProcessor.is_likely_heading("EMPLOYEE BENEFITS") is True
    assert DocumentProcessor.is_likely_heading("This is a regular long sentence explaining details about something.") is False


def test_extract_text_from_pdf(sample_pdf_bytes):
    processor = DocumentProcessor()
    pages = processor.extract_text_from_pdf(sample_pdf_bytes)
    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert "Welcome to Nexain" in pages[0]["text"]
    assert pages[1]["page"] == 2
    assert "Working Hours" in pages[1]["text"]


def test_extract_text_from_invalid_pdf():
    processor = DocumentProcessor()
    with pytest.raises(DocumentProcessingException):
        processor.extract_text_from_pdf(b"not a valid pdf binary content")


def test_chunk_document_heading_aware(sample_pdf_bytes):
    processor = DocumentProcessor()
    pages = processor.extract_text_from_pdf(sample_pdf_bytes)
    chunks = processor.chunk_document(pages, target_chunk_chars=500, overlap_chars=50)

    assert len(chunks) >= 1
    sections = [c["section"] for c in chunks]
    assert any("Welcome to Nexain" in s or "Working Hours" in s for s in sections)
    assert all("page" in c for c in chunks)
    assert all("content" in c for c in chunks)


@pytest.mark.asyncio
async def test_upload_and_process_version_end_to_end(isolated_processor, sample_pdf_bytes):
    # 1. Setup logical document
    doc = Document(
        id="leave-policy",
        title="Leave Policy",
        document_type=DocumentType.POLICY,
        department="People Operations",
        location="Jakarta",
    )
    await isolated_processor.firestore_repo.create_document(doc)

    # 2. Upload version
    version = await isolated_processor.upload_version(
        document_id="leave-policy",
        pdf_bytes=sample_pdf_bytes,
        version_str="1.0",
        uploaded_by="hr_test",
    )
    assert version.document_id == "leave-policy"
    assert version.version == "1.0"
    assert version.status == VersionStatus.PROCESSING
    assert version.file_hash is not None
    assert version.content_hash is not None

    # 3. Process version
    result = await isolated_processor.process_version(version.id)
    assert result.status == VersionStatus.REVIEW
    assert result.chunk_count > 0
    assert len(result.chunks) == result.chunk_count

    # Check stored chunks in Firestore repo
    stored_chunks = await isolated_processor.firestore_repo.get_chunks_by_version(version.id)
    assert len(stored_chunks) == result.chunk_count
    assert stored_chunks[0].document_id == "leave-policy"
    assert stored_chunks[0].version_id == version.id
    assert stored_chunks[0].embedding is not None
    assert len(stored_chunks[0].embedding) == 768


@pytest.mark.asyncio
async def test_duplicate_file_detection(isolated_processor, sample_pdf_bytes):
    doc = Document(
        id="handbook",
        title="Employee Handbook",
        document_type=DocumentType.HANDBOOK,
        department="People Operations",
    )
    await isolated_processor.firestore_repo.create_document(doc)

    # Upload first time
    await isolated_processor.upload_version(
        document_id="handbook",
        pdf_bytes=sample_pdf_bytes,
        version_str="1.0",
    )

    # Upload exact same PDF bytes again -> should raise DuplicateDocumentException
    with pytest.raises(DuplicateDocumentException) as exc_info:
        await isolated_processor.upload_version(
            document_id="handbook",
            pdf_bytes=sample_pdf_bytes,
            version_str="1.1",
        )
    assert "Duplicate file detected" in str(exc_info.value)


@pytest.mark.asyncio
async def test_duplicate_content_detection(isolated_processor):
    doc = Document(
        id="policy-doc",
        title="Security Policy",
        document_type=DocumentType.IT,
        department="Engineering",
    )
    await isolated_processor.firestore_repo.create_document(doc)

    text_pages = ["1. IT Security\nAll employees must use 2FA."]
    pdf_bytes_v1 = make_test_pdf(text_pages)
    await isolated_processor.upload_version(
        document_id="policy-doc",
        pdf_bytes=pdf_bytes_v1,
        version_str="1.0",
    )

    # Create a slightly different PDF binary structure that contains the exact same text
    pdf_bytes_v2 = make_test_pdf(text_pages) + b"\n% extra trailing comment to alter file hash"

    # File hashes will differ:
    assert DocumentProcessor.calculate_bytes_hash(pdf_bytes_v1) != DocumentProcessor.calculate_bytes_hash(pdf_bytes_v2)

    # But content hash is identical -> must be rejected
    with pytest.raises(DuplicateDocumentException) as exc_info:
        await isolated_processor.upload_version(
            document_id="policy-doc",
            pdf_bytes=pdf_bytes_v2,
            version_str="1.1",
        )
    assert "Duplicate content detected" in str(exc_info.value)
