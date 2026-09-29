import httpx
import pytest

from app.core.config import Settings
from app.main import app
from app.models.document import DocumentType, VersionStatus
from app.repositories.firestore import FirestoreRepository, get_firestore_repository
from app.repositories.storage import StorageRepository, get_storage_repository
from app.services.document_processor import DocumentProcessor, get_document_processor
from tests.conftest import make_test_pdf


@pytest.fixture(autouse=True)
def clean_app_dependencies(tmp_path):
    """Ensure clean isolated repos for each API test."""
    settings = Settings(local_storage_dir=str(tmp_path / "storage"))
    clean_firestore = FirestoreRepository(settings=settings)
    clean_storage = StorageRepository(settings=settings)
    clean_processor = DocumentProcessor(
        firestore_repo=clean_firestore,
        storage_repo=clean_storage,
        settings=settings,
    )

    app.dependency_overrides[get_firestore_repository] = lambda: clean_firestore
    app.dependency_overrides[get_storage_repository] = lambda: clean_storage
    app.dependency_overrides[get_document_processor] = lambda: clean_processor

    yield

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_create_and_list_documents():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 1. Create document
        res = await client.post(
            "/api/v1/documents",
            json={
                "title": "Leave Policy",
                "document_type": "POLICY",
                "department": "People Operations",
                "location": "Jakarta",
            },
        )
        assert res.status_code == 201
        data = res.json()
        assert data["title"] == "Leave Policy"
        assert data["id"] == "leave-policy"
        assert data["department"] == "People Operations"

        # 2. List documents
        res_list = await client.get("/api/v1/documents")
        assert res_list.status_code == 200
        docs = res_list.json()
        assert len(docs) == 1
        assert docs[0]["id"] == "leave-policy"

        # 3. Filter documents by department
        res_filtered = await client.get("/api/v1/documents?department=People Operations")
        assert res_filtered.status_code == 200
        assert len(res_filtered.json()) == 1

        res_none = await client.get("/api/v1/documents?department=NonExistent")
        assert res_none.status_code == 200
        assert len(res_none.json()) == 0


@pytest.mark.asyncio
async def test_get_document_by_id():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Create
        await client.post(
            "/api/v1/documents",
            json={
                "title": "Employee Handbook",
                "document_type": "HANDBOOK",
                "department": "People Operations",
            },
        )

        res = await client.get("/api/v1/documents/employee-handbook")
        assert res.status_code == 200
        assert res.json()["title"] == "Employee Handbook"

        res_not_found = await client.get("/api/v1/documents/missing-doc")
        assert res_not_found.status_code == 404
        assert res_not_found.json()["error"]["code"] == "RESOURCE_NOT_FOUND"


@pytest.mark.asyncio
async def test_upload_and_process_pdf_lifecycle(sample_pdf_bytes):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 1. Create document
        doc_res = await client.post(
            "/api/v1/documents",
            json={
                "title": "Laptop Request Procedure",
                "document_type": "PROCEDURE",
                "department": "Engineering",
                "location": "Jakarta",
            },
        )
        doc_id = doc_res.json()["id"]

        # 2. Upload PDF version
        upload_res = await client.post(
            f"/api/v1/documents/{doc_id}/versions",
            files={"file": ("laptop_proc.pdf", sample_pdf_bytes, "application/pdf")},
            data={"version": "1.0"},
        )
        assert upload_res.status_code == 201
        upload_data = upload_res.json()
        version_id = upload_data["version_id"]
        assert upload_data["status"] == "PROCESSING"
        assert upload_data["file_hash"] is not None

        # 3. Process Version into chunks
        process_res = await client.post(f"/api/v1/versions/{version_id}/process")
        assert process_res.status_code == 200
        process_data = process_res.json()
        assert process_data["status"] == "REVIEW"
        assert process_data["chunk_count"] > 0

        # 4. Get chunks
        chunks_res = await client.get(f"/api/v1/versions/{version_id}/chunks")
        assert chunks_res.status_code == 200
        chunks = chunks_res.json()
        assert len(chunks) == process_data["chunk_count"]
        assert chunks[0]["document_id"] == doc_id
        assert chunks[0]["page"] in (1, 2)

        # 5. Publish version
        pub_res = await client.post(f"/api/v1/versions/{version_id}/publish")
        assert pub_res.status_code == 200
        assert pub_res.json()["status"] == "PUBLISHED"

        # Check document current_version_id
        doc_after = await client.get(f"/api/v1/documents/{doc_id}")
        assert doc_after.json()["current_version_id"] == version_id


@pytest.mark.asyncio
async def test_duplicate_upload_rejection_api(sample_pdf_bytes):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Create doc
        await client.post(
            "/api/v1/documents",
            json={"title": "Test Doc", "document_type": "POLICY", "department": "HR"},
        )

        # Upload v1
        res1 = await client.post(
            "/api/v1/documents/test-doc/versions",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert res1.status_code == 201

        # Upload exact same file again -> expect 409 DUPLICATE_DOCUMENT
        res2 = await client.post(
            "/api/v1/documents/test-doc/versions",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert res2.status_code == 409
        err = res2.json()
        assert err["error"]["code"] == "DUPLICATE_DOCUMENT"
        assert "Duplicate file detected" in err["error"]["message"]


@pytest.mark.asyncio
async def test_reject_version_api(sample_pdf_bytes):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/api/v1/documents",
            json={"title": "Draft Doc", "document_type": "GUIDE", "department": "HR"},
        )
        up = await client.post(
            "/api/v1/documents/draft-doc/versions",
            files={"file": ("draft.pdf", sample_pdf_bytes, "application/pdf")},
        )
        version_id = up.json()["version_id"]

        reject_res = await client.post(f"/api/v1/versions/{version_id}/reject")
        assert reject_res.status_code == 200
        assert reject_res.json()["status"] == "REJECTED"
