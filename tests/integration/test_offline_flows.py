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
from app.models.document import VersionStatus
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


async def _create_profile(client, user_id="user_001", role="Backend Engineer", department="Engineering"):
    res = await client.post("/api/v1/profile", headers={"X-User-Id": user_id}, json={
        "name": "Aji", "role": role, "department": department, "location": "Jakarta"})
    assert res.status_code == 201, res.text
    return res.json()


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

        doc = (await client.get(f"/api/v1/documents/{doc_id}", headers=HR)).json()
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
        await _create_profile(client)
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
        await _create_profile(client)
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

        all_docs = (await client.get("/api/v1/documents", headers=HR)).json()
        assert len(all_docs) == 2

        # Employees only see documents that have a published version.
        employee_docs = (await client.get("/api/v1/documents", headers=EMP)).json()
        assert [d["id"] for d in employee_docs] == ["handbook"]

        filtered = (await client.get("/api/v1/documents?location=Jakarta", headers=HR)).json()
        assert all(d["location"] == "Jakarta" for d in filtered)

        published = (await client.get("/api/v1/documents?status=PUBLISHED")).json()
        assert any(d["id"] == "handbook" for d in published)


# ---------------------------------------------------------------- review fixes


class CitingVertex:
    """Fake Gemini that records requests and answers citing only [Source 1]."""

    def __init__(self):
        self.requests = []

    async def generate_chat(self, request):  # noqa: ANN001
        from app.schemas.chat import ChatResponse

        self.requests.append(request)
        return ChatResponse(response="You can carry over up to five days [Source 1].", model="fake")


@pytest.mark.asyncio
async def test_chat_works_without_profile(api):
    """HR admins have no onboarding profile; chat must still answer, not 404."""
    async with _client() as client:
        await _ingest(client, "Leave Policy", [
            "1. Annual Leave\nEmployees receive twelve days of annual leave each calendar year.",
        ])
        res = await client.post("/api/v1/chat", headers=HR, json={
            "user_id": "hr_admin_001", "message": "How many days of annual leave do I get?"})
        assert res.status_code == 200, res.text
        assert res.json()["grounded"] is True

        res = await client.post("/api/v1/chat", headers=HR, json={
            "user_id": "hr_admin_001", "message": "What are my tasks?"})
        assert res.status_code == 200, res.text
        assert res.json()["intent"] == "MY_ONBOARDING"
        assert "profile" in res.json()["answer"].lower()


@pytest.mark.asyncio
async def test_chat_refuses_other_user_id(api):
    async with _client() as client:
        res = await client.post("/api/v1/chat", headers=EMP, json={
            "user_id": "user_002", "message": "What are my tasks?"})
        assert res.status_code == 403


@pytest.mark.asyncio
async def test_follow_up_uses_history_and_only_cited_sources(api):
    vertex = CitingVertex()
    app.dependency_overrides[get_rag_engine] = lambda: RAGEngine(
        retrieval_service=api["retrieval"], vertex_service=vertex)
    async with _client() as client:
        await _ingest(client, "Leave Policy", [
            "1. Annual Leave\nEmployees receive twelve days of annual leave. Unused annual leave days "
            "can be carried over to next year, up to five days.",
        ])
        await _ingest(client, "Benefit Guide", [
            "1. Health Plan\nThe health plan covers outpatient visits and annual dental leave checkups.",
        ])

        res = await client.post("/api/v1/chat", headers=EMP, json={
            "user_id": "user_001",
            "message": "And can I carry it over?",
            "history": [
                {"role": "user", "content": "How many days of annual leave do I get?"},
                {"role": "assistant", "content": "You get twelve days of annual leave."},
                {"role": "bogus", "content": "ignored"},
            ],
        })
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["intent"] == "COMPANY_KNOWLEDGE"
        assert body["grounded"] is True
        # [Source N] markers are stripped from the answer text.
        assert body["answer"] == "You can carry over up to five days."
        # Only the cited source is returned, not every retrieved chunk.
        assert [c["document_id"] for c in body["citations"]] == ["leave-policy"]

        # Earlier turns are passed to Gemini with normalized roles.
        sent = vertex.requests[-1]
        assert [(m.role, m.content) for m in sent.history] == [
            ("user", "How many days of annual leave do I get?"),
            ("model", "You get twelve days of annual leave."),
        ]


@pytest.mark.asyncio
async def test_employees_cannot_manage_or_see_unpublished_documents(api):
    spoofed = {"X-User-Id": "user_002", "X-Access-Level": "HR_ADMIN"}
    async with _client() as client:
        # A client-supplied access level no longer grants HR rights.
        res = await client.post("/api/v1/documents", headers=spoofed, json={"title": "Sneaky"})
        assert res.status_code == 403
        assert res.json()["error"]["code"] == "UNAUTHORIZED"

        _, draft_version = await _ingest(client, "Draft Plan", [
            "1. Draft\nThis plan is still being reviewed by HR."], publish=False)
        _, live_version = await _ingest(client, "Live Guide", [
            "1. Welcome\nWelcome to the company, here is the live guide."])

        docs = (await client.get("/api/v1/documents", headers=spoofed)).json()
        assert [d["id"] for d in docs] == ["live-guide"]
        assert all(v["storage_path"] == "" and v["file_hash"] == "" for v in docs[0]["versions"])

        assert (await client.get("/api/v1/documents/draft-plan", headers=EMP)).status_code == 404
        assert (await client.get(f"/api/v1/versions/{draft_version}/chunks", headers=EMP)).status_code == 404
        assert (await client.get(f"/api/v1/versions/{live_version}/chunks", headers=EMP)).status_code == 200
        assert (await client.get(f"/api/v1/versions/{draft_version}/evaluation", headers=EMP)).status_code == 403

        # HR still sees everything, including storage paths.
        hr_docs = (await client.get("/api/v1/documents", headers=HR)).json()
        assert {d["id"] for d in hr_docs} == {"draft-plan", "live-guide"}
        assert all(v["storage_path"] for d in hr_docs for v in d["versions"])


@pytest.mark.asyncio
async def test_publish_supersedes_older_stuck_versions(api):
    async with _client() as client:
        res = await client.post("/api/v1/documents", headers=HR, json={"title": "Leave Policy"})
        doc_id = res.json()["id"]

        # v1 is uploaded but never processed, so it stays in PROCESSING.
        res = await client.post(f"/api/v1/documents/{doc_id}/versions", headers=HR, files={
            "file": ("v1.pdf", make_test_pdf(["1. Annual Leave\nTen days of leave."]), "application/pdf")})
        stuck = res.json()["version_id"]

        res = await client.post(f"/api/v1/documents/{doc_id}/versions", headers=HR, files={
            "file": ("v2.pdf", make_test_pdf(["1. Annual Leave\nTwelve days of leave."]), "application/pdf")})
        live = res.json()["version_id"]
        await client.post(f"/api/v1/versions/{live}/process", headers=HR)
        res = await client.post(f"/api/v1/versions/{live}/publish", headers=HR)
        assert res.status_code == 200, res.text

        doc = (await client.get(f"/api/v1/documents/{doc_id}", headers=HR)).json()
        statuses = {v["id"]: v["status"] for v in doc["versions"]}
        assert statuses == {stuck: "SUPERSEDED", live: "PUBLISHED"}


@pytest.mark.asyncio
async def test_reprocessing_published_version_keeps_it_live(api):
    async with _client() as client:
        doc_id, version_id = await _ingest(client, "Leave Policy", [
            "1. Annual Leave\nEmployees receive twelve days of annual leave."])
        res = await client.post(f"/api/v1/versions/{version_id}/process", headers=HR)
        assert res.status_code == 200, res.text
        assert res.json()["status"] == "PUBLISHED"

        res = await client.post("/api/v1/chat", headers=EMP, json={
            "user_id": "user_001", "message": "How many days of annual leave do I get?"})
        assert res.json()["grounded"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("question", ["what's left for me", "apa tugas saya hari ini?", "my checklist"])
async def test_my_tasks_question_lists_pending_tasks(api, question):
    async with _client() as client:
        await _create_profile(client)
        tasks = (await client.get("/api/v1/tasks", headers=EMP)).json()
        await client.post(f"/api/v1/tasks/{tasks[0]['id']}/complete", headers=EMP)

        res = await client.post("/api/v1/chat", headers=EMP, json={"user_id": "user_001", "message": question})
        body = res.json()
        assert body["intent"] == "MY_ONBOARDING", body
        assert "You still have 6 tasks left" in body["answer"]
        for task in tasks[1:]:
            assert f"- Day {task['day']}: {task['title']}" in body["answer"]
        assert tasks[0]["title"] not in body["answer"]


# ---------------------------------------------------------------- reconcile


async def _upload(client, doc_id, pages, name):
    res = await client.post(f"/api/v1/documents/{doc_id}/versions", headers=HR, files={
        "file": (name, make_test_pdf(pages), "application/pdf")})
    assert res.status_code == 201, res.text
    return res.json()["version_id"]


@pytest.mark.asyncio
async def test_reconcile_repairs_legacy_lifecycle_data(api):
    firestore = api["firestore"]
    header = "DayOne AI - Internal HR Knowledge Base"
    async with _client() as client:
        # Leave policy as it is in production: v1 stuck in PROCESSING, v2 published,
        # current_version_id null, and chunks carrying the page header as section.
        doc_id = (await client.post("/api/v1/documents", headers=HR, json={"title": "Leave Policy"})).json()["id"]
        stuck = await _upload(client, doc_id, ["1. Annual Leave\nTen days."], "v1.pdf")
        live = await _upload(client, doc_id, [
            f"{header}\nPage 1\n1. Annual Leave\nEmployees receive twelve days of annual leave.",
            f"{header}\nPage 2\n2. Sick Leave\nSick leave needs a doctor's note after two days.",
        ], "v2.pdf")
        await client.post(f"/api/v1/versions/{live}/process", headers=HR)
        v = await firestore.get_document_version(live)
        v.status = VersionStatus.PUBLISHED
        await firestore.update_document_version(v)
        for chunk in await firestore.get_chunks_by_version(live):
            chunk.section = header
            await firestore.save_chunks([chunk])

        # A second document with two PUBLISHED versions and a stale current_version_id.
        _, old_live = await _ingest(client, "Benefit Guide", ["1. Health\nOutpatient visits are covered."])
        new_live = await _upload(client, "benefit-guide", ["1. Health\nOutpatient and dental visits are covered."], "b2.pdf")
        await client.post(f"/api/v1/versions/{new_live}/process", headers=HR)
        v = await firestore.get_document_version(new_live)
        v.status = VersionStatus.PUBLISHED
        await firestore.update_document_version(v)

        # Employees cannot run it.
        res = await client.post("/api/v1/admin/knowledge-base/reconcile", headers=EMP)
        assert res.status_code == 403

        # Dry run reports but changes nothing.
        res = await client.post("/api/v1/admin/knowledge-base/reconcile", headers=HR)
        assert res.status_code == 200, res.text
        dry = res.json()
        assert dry["dry_run"] is True
        assert dry["versions_superseded"] == 2
        assert dry["current_version_ids_fixed"] == 2
        assert (await firestore.get_document_version(stuck)).status == VersionStatus.PROCESSING

        res = await client.post("/api/v1/admin/knowledge-base/reconcile?dry_run=false", headers=HR)
        assert res.status_code == 200, res.text
        report = res.json()
        assert report["versions_reprocessed"] == 2 and report["reprocess_errors"] == 0

        leave = (await client.get(f"/api/v1/documents/{doc_id}", headers=HR)).json()
        assert leave["current_version_id"] == live
        assert {v["id"]: v["status"] for v in leave["versions"]} == {stuck: "SUPERSEDED", live: "PUBLISHED"}

        benefit = (await client.get("/api/v1/documents/benefit-guide", headers=HR)).json()
        assert benefit["current_version_id"] == new_live
        assert {v["id"]: v["status"] for v in benefit["versions"]} == {old_live: "SUPERSEDED", new_live: "PUBLISHED"}

        # Live chunks were rebuilt with real headings and no page furniture.
        chunks = await firestore.get_chunks_by_version(live)
        assert [c.section for c in chunks] == ["1. Annual Leave", "2. Sick Leave"]
        assert all(header not in c.content and "Page " not in c.content for c in chunks)
        leave_report = next(d for d in report["documents"] if d["document_id"] == doc_id)
        assert leave_report["reprocessed"][0]["sections"] == ["1. Annual Leave", "2. Sick Leave"]

        # Running again finds nothing left to fix.
        again = (await client.post(
            "/api/v1/admin/knowledge-base/reconcile?dry_run=false&reprocess=false", headers=HR)).json()
        assert again["documents_changed"] == 0
