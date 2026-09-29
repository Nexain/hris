"""End-to-end integration tests — fully offline (no Google Cloud, no AI).

These exercise the complete request stack (FastAPI routes -> services ->
repositories) with isolated in-memory Firestore, local-disk storage, offline
lexical embeddings, and a stubbed Vertex client that forces the RAG engine's
deterministic extractive answer path.
"""

import httpx
import pytest

from app.core.config import Settings
from app.main import app
from app.repositories.firestore import FirestoreRepository, get_firestore_repository
from app.repositories.storage import StorageRepository, get_storage_repository
from app.services.document_processor import DocumentProcessor, get_document_processor
from app.services.embeddings import EmbeddingService
from app.services.gemini import RAGEngine, get_rag_engine
from app.services.onboarding import OnboardingService, get_onboarding_service
from app.services.retrieval import RetrievalService, get_retrieval_service
from app.services.vertex_service import get_vertex_service
from tests.conftest import make_test_pdf


HR = {"X-Access-Level": "HR_ADMIN", "X-User-Id": "hr_admin_001"}
EMP = {"X-User-Id": "user_001"}


class OfflineVertex:
    """Fail fast so RAGEngine uses its grounded extractive fallback (no network)."""

    async def generate_chat(self, request):  # noqa: ANN001
        raise RuntimeError("AI is disabled in offline integration tests")

    async def generate_chat_stream(self, request):  # noqa: ANN001
        raise RuntimeError("AI is disabled in offline integration tests")
        yield ""  # pragma: no cover


@pytest.fixture
def api(tmp_path):
    """Isolated app with in-memory Firestore and local-disk storage."""
    settings = Settings(local_storage_dir=str(tmp_path / "storage"))
    firestore = FirestoreRepository(settings=settings)
    storage = StorageRepository(settings=settings)
    embeddings = EmbeddingService(settings=settings)
    processor = DocumentProcessor(
        firestore_repo=firestore,
        storage_repo=storage,
        embedding_service=embeddings,
        settings=settings,
    )
    retrieval = RetrievalService(firestore_repo=firestore, embedding_service=embeddings)
    onboarding = OnboardingService(firestore_repo=firestore)
    vertex = OfflineVertex()
    rag = RAGEngine(retrieval_service=retrieval, vertex_service=vertex, settings=settings)

    app.dependency_overrides[get_firestore_repository] = lambda: firestore
    app.dependency_overrides[get_storage_repository] = lambda: storage
    app.dependency_overrides[get_document_processor] = lambda: processor
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval
    app.dependency_overrides[get_onboarding_service] = lambda: onboarding
    app.dependency_overrides[get_rag_engine] = lambda: rag
    app.dependency_overrides[get_vertex_service] = lambda: vertex

    yield {"firestore": firestore, "processor": processor, "retrieval": retrieval}

    app.dependency_overrides.clear()


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def _ingest(client, title, pages, publish=True):
    """Create -> upload -> process -> (optionally) publish a document."""
    res = await client.post("/api/v1/documents", headers=HR, json={"title": title, "document_type": "PROCEDURE"})
    assert res.status_code == 201, res.text
    doc_id = res.json()["id"]

    pdf = make_test_pdf(pages)
    res = await client.post(
        f"/api/v1/documents/{doc_id}/versions",
        headers=HR,
        files={"file": (f"{doc_id}.pdf", pdf, "application/pdf")},
    )
    assert res.status_code == 201, res.text
    version_id = res.json()["version_id"]

    res = await client.post(f"/api/v1/versions/{version_id}/process", headers=HR)
    assert res.status_code == 200, res.text

    if publish:
        res = await client.post(f"/api/v1/versions/{version_id}/publish", headers=HR)
        assert res.status_code == 200, res.text

    return doc_id, version_id


# ---------------------------------------------------------------- full flow


@pytest.mark.asyncio
async def test_full_ingestion_to_grounded_chat(api):
    async with _client() as client:
        doc_id, version_id = await _ingest(client, "Laptop Request Procedure", [
            "1. Request Process\nTo request a laptop, submit a hardware request form through the IT portal.",
        ])

        res = await client.post("/api/v1/chat", headers=EMP, json={
            "user_id": "user_001", "message": "How do I request a laptop?"})
        assert res.status_code == 200, res.text
        body = res.json()

        assert body["intent"] == "COMPANY_KNOWLEDGE"
        assert body["grounded"] is True
        assert body["escalation_required"] is False
        assert len(body["citations"]) == 1
        citation = body["citations"][0]
        assert citation["document_id"] == doc_id
        assert citation["document_title"] == "Laptop Request Procedure"
        assert citation["version"] == "1.0"
        assert citation["page"] == 1
        assert "laptop" in body["answer"].lower()


@pytest.mark.asyncio
async def test_unpublished_version_is_not_retrievable(api):
    async with _client() as client:
        await _ingest(client, "Draft Rollout", [
            "1. Confidential Rollout\nThis confidential rollout plan must not reach employees yet.",
        ], publish=False)

        res = await client.post("/api/v1/chat", headers=EMP, json={
            "user_id": "user_001", "message": "What is the confidential rollout plan?"})
        body = res.json()

        assert body["grounded"] is False
        assert body["escalation_required"] is True
        assert body["citations"] == []


# ---------------------------------------------------------------- duplicates


@pytest.mark.asyncio
async def test_duplicate_file_and_content_detection(api):
    async with _client() as client:
        doc_id, _ = await _ingest(client, "Leave Policy", [
            "1. Annual Leave\nEmployees receive twelve days of annual leave.",
        ])

        same_pdf = make_test_pdf(["1. Annual Leave\nEmployees receive twelve days of annual leave."])
        res = await client.post(
            f"/api/v1/documents/{doc_id}/versions",
            headers=HR,
            files={"file": ("dup.pdf", same_pdf, "application/pdf")},
        )
        assert res.status_code == 409
        assert res.json()["error"]["code"] == "DUPLICATE_DOCUMENT"

        # Different bytes, identical normalized text -> content duplicate.
        content_dup = make_test_pdf(["1. Annual Leave\nEmployees receive  twelve   days of annual leave."])
        res = await client.post(
            f"/api/v1/documents/{doc_id}/versions",
            headers=HR,
            files={"file": ("content-dup.pdf", content_dup, "application/pdf")},
        )
        assert res.status_code == 409
        assert res.json()["error"]["code"] == "DUPLICATE_DOCUMENT"


# ---------------------------------------------------------------- lifecycle


@pytest.mark.asyncio
async def test_publish_requires_review_state(api):
    async with _client() as client:
        res = await client.post("/api/v1/documents", headers=HR, json={"title": "Expense Policy"})
        doc_id = res.json()["id"]
        pdf = make_test_pdf(["1. Reimbursement\nSubmit receipts within thirty days."])
        res = await client.post(
            f"/api/v1/documents/{doc_id}/versions",
            headers=HR,
            files={"file": ("exp.pdf", pdf, "application/pdf")},
        )
        version_id = res.json()["version_id"]

        # Still PROCESSING -> publishing must be rejected.
        res = await client.post(f"/api/v1/versions/{version_id}/publish", headers=HR)
        assert res.status_code == 409
        assert res.json()["error"]["code"] == "INVALID_STATE_TRANSITION"

        # After processing (REVIEW) -> publish succeeds.
        await client.post(f"/api/v1/versions/{version_id}/process", headers=HR)
        res = await client.post(f"/api/v1/versions/{version_id}/publish", headers=HR)
        assert res.status_code == 200
        assert res.json()["status"] == "PUBLISHED"

        # A published version can no longer be rejected.
        res = await client.post(f"/api/v1/versions/{version_id}/reject", headers=HR)
        assert res.status_code == 409
        assert res.json()["error"]["code"] == "INVALID_STATE_TRANSITION"


@pytest.mark.asyncio
async def test_new_version_supersedes_previous(api):
    async with _client() as client:
        doc_id, v1 = await _ingest(client, "Security Guide", [
            "1. Passwords\nUse a strong unique password for every account.",
        ])
        # v2 with distinct content so it is not a duplicate.
        pdf = make_test_pdf(["1. Passwords\nEnable two factor authentication on all accounts."])
        res = await client.post(
            f"/api/v1/documents/{doc_id}/versions",
            headers=HR,
            files={"file": ("v2.pdf", pdf, "application/pdf")},
        )
        v2 = res.json()["version_id"]
        await client.post(f"/api/v1/versions/{v2}/process", headers=HR)
        await client.post(f"/api/v1/versions/{v2}/publish", headers=HR)

        doc = (await client.get(f"/api/v1/documents/{doc_id}")).json()
        assert doc["current_version_id"] == v2
        statuses = {v["id"]: v["status"] for v in doc["versions"]}
        assert statuses[v1] == "SUPERSEDED"
        assert statuses[v2] == "PUBLISHED"

        res = await client.post("/api/v1/chat", headers=EMP, json={
            "user_id": "user_001", "message": "What does the guide say about two factor authentication?"})
        body = res.json()
        assert body["grounded"] is True
        assert body["citations"][0]["version"] == "2.0"
        assert "two factor" in body["answer"].lower()


# ---------------------------------------------------------------- evaluation


@pytest.mark.asyncio
async def test_evaluation_questions_generated_and_persist(api):
    firestore = api["firestore"]
    async with _client() as client:
        doc_id, v1 = await _ingest(client, "Remote Work Policy", [
            "1. Eligibility\nEmployees may work remotely three days per week.",
            "2. Equipment\nA laptop and secure VPN access are provided.",
        ])

        questions = await firestore.get_evaluation_questions(doc_id)
        assert questions, "processing must generate evaluation questions"
        original_ids = sorted(q.id for q in questions)

        # A second version must reuse the golden set (no duplicates / overwrite).
        pdf = make_test_pdf(["1. Eligibility\nEmployees may work remotely two days per week."])
        res = await client.post(
            f"/api/v1/documents/{doc_id}/versions",
            headers=HR,
            files={"file": ("rw2.pdf", pdf, "application/pdf")},
        )
        v2 = res.json()["version_id"]
        await client.post(f"/api/v1/versions/{v2}/process", headers=HR)

        assert sorted(q.id for q in await firestore.get_evaluation_questions(doc_id)) == original_ids

        # Regression evaluation for the new version.
        res = await client.get(f"/api/v1/versions/{v2}/evaluation", headers=HR)
        assert res.status_code == 200
        report = res.json()
        assert report["total"] == len(original_ids)
        assert report["hit_rate"] >= 0.8
        assert report["passed"] is True


# ---------------------------------------------------------------- onboarding


@pytest.mark.asyncio
async def test_onboarding_profile_tasks_and_progress(api):
    async with _client() as client:
        profile = (await client.get("/api/v1/profile", headers=EMP)).json()
        assert profile["user_id"] == "user_001"
        assert profile["role"] == "Backend Engineer"
        assert profile["department"] == "Engineering"

        tasks = (await client.get("/api/v1/tasks", headers=EMP)).json()
        assert len(tasks) == 7

        res = await client.post(f"/api/v1/tasks/{tasks[0]['id']}/complete", headers=EMP)
        assert res.status_code == 200
        assert res.json()["status"] == "COMPLETED"

        progress = (await client.get("/api/v1/onboarding/progress", headers=EMP)).json()
        assert progress["total"] == 7
        assert progress["completed"] == 1
        assert progress["remaining"] == 6
        assert progress["progress_percent"] == round((1 / 7) * 100, 2)

        # Next-action intent routes to onboarding, not RAG.
        res = await client.post("/api/v1/chat", headers=EMP, json={
            "user_id": "user_001", "message": "What should I do next in my onboarding?"})
        body = res.json()
        assert body["intent"] == "MY_ONBOARDING"
        assert body["citations"] == []


# ---------------------------------------------------------------- security


@pytest.mark.asyncio
async def test_security_boundaries(api):
    async with _client() as client:
        tasks = (await client.get("/api/v1/tasks", headers=EMP)).json()

        # Another employee cannot complete user_001's task.
        res = await client.post(
            f"/api/v1/tasks/{tasks[0]['id']}/complete",
            headers={"X-User-Id": "user_002"},
        )
        assert res.status_code == 403
        assert res.json()["error"]["code"] == "UNAUTHORIZED"

        # Sensitive request is rejected and escalated, never grounded.
        res = await client.post("/api/v1/chat", headers=EMP, json={
            "user_id": "user_001", "message": "How much does my colleague earn?"})
        body = res.json()
        assert body["intent"] == "OUT_OF_SCOPE"
        assert body["grounded"] is False
        assert body["escalation_required"] is True
        assert body["citations"] == []

        # Out-of-scope request is refused.
        res = await client.post("/api/v1/chat", headers=EMP, json={
            "user_id": "user_001", "message": "Tell me a joke about cats"})
        assert res.json()["intent"] == "OUT_OF_SCOPE"


# ---------------------------------------------------------------- listing


@pytest.mark.asyncio
async def test_list_and_filter_documents(api):
    async with _client() as client:
        await _ingest(client, "Handbook", ["1. Welcome\nWelcome to the company."])
        res = await client.post("/api/v1/documents", headers=HR, json={
            "title": "Jakarta Benefits", "department": "People Operations", "location": "Jakarta"})
        assert res.status_code == 201

        all_docs = (await client.get("/api/v1/documents")).json()
        assert len(all_docs) == 2

        filtered = (await client.get("/api/v1/documents?location=Jakarta")).json()
        assert all(d["location"] == "Jakarta" for d in filtered)

        published = (await client.get("/api/v1/documents?status=PUBLISHED")).json()
        assert any(d["id"] == "handbook" for d in published)
