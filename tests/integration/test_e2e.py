"""End-to-end integration test — full employee/HR journey, fully offline.

Cloud is stubbed (in-memory Firestore + local-disk storage) and the LLM is
stubbed (a deterministic generator that answers from the retrieved context).
This exercises the complete request stack without any Google Cloud or AI calls.
"""

import re
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio

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

FALLBACK = (
    "I couldn't find reliable information about that in the available company "
    "documentation. Please contact People Operations for assistance."
)


class StubGemini:
    """Offline LLM: answers deterministically from the retrieved RAG context."""

    async def generate_chat(self, request):  # noqa: ANN001
        prompt = request.message
        if "Content:\n" not in prompt:
            return SimpleNamespace(response=FALLBACK)
        after = prompt.split("Content:\n", 1)[1]
        content = after.split("\n---\n", 1)[0].split("\n\nPlease answer", 1)[0].strip()
        return SimpleNamespace(response=f"According to the company documentation: {content}")

    async def generate_chat_stream(self, request):  # noqa: ANN001
        resp = await self.generate_chat(request)
        yield resp.response


class RefusingGemini:
    """Offline LLM that declines to answer, to exercise the fallback path."""

    async def generate_chat(self, request):  # noqa: ANN001
        return SimpleNamespace(response=FALLBACK)

    async def generate_chat_stream(self, request):  # noqa: ANN001
        yield FALLBACK


def _wire(settings, firestore, storage, embeddings, processor, retrieval, onboarding, vertex):
    rag = RAGEngine(retrieval_service=retrieval, vertex_service=vertex, settings=settings)
    app.dependency_overrides[get_firestore_repository] = lambda: firestore
    app.dependency_overrides[get_storage_repository] = lambda: storage
    app.dependency_overrides[get_document_processor] = lambda: processor
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval
    app.dependency_overrides[get_onboarding_service] = lambda: onboarding
    app.dependency_overrides[get_rag_engine] = lambda: rag
    app.dependency_overrides[get_vertex_service] = lambda: vertex


@pytest_asyncio.fixture
async def e2e(tmp_path):
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
    vertex = StubGemini()
    _wire(settings, firestore, storage, embeddings, processor, retrieval, onboarding, vertex)

    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    state = {"client": client, "firestore": firestore, "settings": settings, "embeddings": embeddings,
             "processor": processor, "retrieval": retrieval, "onboarding": onboarding}
    try:
        yield state
    finally:
        await client.aclose()
        app.dependency_overrides.clear()


async def _create_upload_process(client, title, pages):
    res = await client.post("/api/v1/documents", headers=HR, json={"title": title})
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

    return doc_id, version_id


@pytest.mark.asyncio
async def test_end_to_end_journey(e2e):
    client = e2e["client"]
    firestore = e2e["firestore"]

    # 0. Health.
    res = await client.get("/api/v1/health")
    assert res.status_code == 200 and res.json()["status"] == "ok"

    # 1. Empty knowledge base -> safe fallback (no hallucination).
    res = await client.post("/api/v1/chat", headers=EMP, json={
        "user_id": "user_001", "message": "How do I request a laptop?"})
    body = res.json()
    assert body["grounded"] is False
    assert body["escalation_required"] is True
    assert body["citations"] == []

    # 2. HR ingests a document.
    doc_id, v1 = await _create_upload_process(client, "Laptop Request Procedure", [
        "1. Request Process\nTo request a laptop, submit a hardware request form through the IT portal.",
        "2. Approval\nYour manager approves the hardware request within two business days.",
    ])

    # 3. Duplicate file is rejected.
    dup = make_test_pdf([
        "1. Request Process\nTo request a laptop, submit a hardware request form through the IT portal.",
        "2. Approval\nYour manager approves the hardware request within two business days.",
    ])
    res = await client.post(
        f"/api/v1/documents/{doc_id}/versions",
        headers=HR,
        files={"file": ("dup.pdf", dup, "application/pdf")},
    )
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "DUPLICATE_DOCUMENT"

    # 4. Processing generated chunks + evaluation questions.
    chunks = await firestore.get_chunks_by_version(v1)
    assert chunks and all(c.embedding for c in chunks)
    questions = await firestore.get_evaluation_questions(doc_id)
    assert questions, "processing must generate evaluation questions"

    # 5. Pre-publish regression evaluation (chunks scored directly).
    res = await client.get(f"/api/v1/versions/{v1}/evaluation", headers=HR)
    assert res.status_code == 200
    assert res.json()["hit_rate"] >= 0.8

    # 6. Publish (REVIEW -> PUBLISHED).
    res = await client.post(f"/api/v1/versions/{v1}/publish", headers=HR)
    assert res.status_code == 200 and res.json()["status"] == "PUBLISHED"

    # 7. Employee company question -> grounded answer with citation.
    res = await client.post("/api/v1/chat", headers=EMP, json={
        "user_id": "user_001", "message": "How do I request a laptop?"})
    body = res.json()
    assert body["intent"] == "COMPANY_KNOWLEDGE"
    assert body["grounded"] is True
    assert len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["document_id"] == doc_id
    assert citation["document_title"] == "Laptop Request Procedure"
    assert citation["version"] == "1.0"
    assert citation["page"] == 1
    assert "laptop" in body["answer"].lower()

    # 8. Onboarding intents.
    res = await client.post("/api/v1/chat", headers=EMP, json={
        "user_id": "user_001", "message": "What should I do next in my onboarding?"})
    assert res.json()["intent"] == "MY_ONBOARDING"

    res = await client.post("/api/v1/chat", headers=EMP, json={
        "user_id": "user_001", "message": "What is the onboarding process?"})
    assert res.json()["intent"] == "GENERAL_ONBOARDING"

    # 9. Personalized tasks + progress.
    tasks = (await client.get("/api/v1/tasks", headers=EMP)).json()
    assert len(tasks) == 7
    res = await client.post(f"/api/v1/tasks/{tasks[0]['id']}/complete", headers=EMP)
    assert res.json()["status"] == "COMPLETED"
    progress = (await client.get("/api/v1/onboarding/progress", headers=EMP)).json()
    assert progress == {"total": 7, "completed": 1, "remaining": 6, "progress_percent": round(100 / 7, 2)}

    # 10. HR uploads v2 (distinct content), processes and publishes -> supersedes v1.
    pdf2 = make_test_pdf(["1. Request Process\nTo request a laptop, use the equipment portal with manager sign-off."])
    res = await client.post(
        f"/api/v1/documents/{doc_id}/versions",
        headers=HR,
        files={"file": ("v2.pdf", pdf2, "application/pdf")},
    )
    v2 = res.json()["version_id"]
    await client.post(f"/api/v1/versions/{v2}/process", headers=HR)
    await client.post(f"/api/v1/versions/{v2}/publish", headers=HR)

    doc = (await client.get(f"/api/v1/documents/{doc_id}")).json()
    statuses = {v["id"]: v["status"] for v in doc["versions"]}
    assert statuses[v1] == "SUPERSEDED"
    assert statuses[v2] == "PUBLISHED"
    assert doc["current_version_id"] == v2

    # 11. Employee now gets the v2 answer.
    res = await client.post("/api/v1/chat", headers=EMP, json={
        "user_id": "user_001", "message": "How do I request a laptop?"})
    body = res.json()
    assert body["grounded"] is True
    assert body["citations"][0]["version"] == "2.0"
    assert "portal" in body["answer"].lower()

    # 12. Golden questions persisted across versions (unchanged set).
    assert [q.id for q in await firestore.get_evaluation_questions(doc_id)] == [q.id for q in questions]

    # 13. Regression evaluation on the new version.
    reg = (await client.get(f"/api/v1/versions/{v2}/evaluation", headers=HR)).json()
    assert reg["passed"] is True and reg["hit_rate"] >= 0.8

    # 14. Audit trail written.
    actions = {entry["action"] for entry in firestore._mem_audit_logs}
    assert {"CREATE_DOCUMENT", "UPLOAD_VERSION", "PROCESS_VERSION", "PUBLISH_VERSION"} <= actions

    # 15. Security boundaries.
    res = await client.post(
        f"/api/v1/tasks/{tasks[0]['id']}/complete", headers={"X-User-Id": "user_002"})
    assert res.status_code == 403 and res.json()["error"]["code"] == "UNAUTHORIZED"

    res = await client.post("/api/v1/chat", headers=EMP, json={
        "user_id": "user_001", "message": "How much does my colleague earn?"})
    body = res.json()
    assert body["intent"] == "OUT_OF_SCOPE"
    assert body["grounded"] is False and body["escalation_required"] is True


@pytest.mark.asyncio
async def test_llm_declines_then_safe_fallback(e2e):
    """When published context exists but the LLM declines, the API returns fallback."""
    client = e2e["client"]

    _settings = e2e["settings"]
    # Swap in a refusing LLM on the same wired services.
    _wire(
        _settings,
        e2e["firestore"],
        app.dependency_overrides[get_storage_repository](),
        e2e["embeddings"],
        e2e["processor"],
        e2e["retrieval"],
        e2e["onboarding"],
        RefusingGemini(),
    )

    doc_id, v1 = await _create_upload_process(client, "Leave Policy", [
        "1. Annual Leave\nEmployees receive twelve days of annual leave each year.",
    ])
    await client.post(f"/api/v1/versions/{v1}/publish", headers=HR)

    res = await client.post("/api/v1/chat", headers=EMP, json={
        "user_id": "user_001", "message": "How many days of annual leave do I get?"})
    body = res.json()
    assert body["grounded"] is False
    assert body["escalation_required"] is True
    assert body["citations"] == []
