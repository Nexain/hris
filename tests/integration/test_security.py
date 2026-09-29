"""Security boundary tests (spec §13, §16.3).

These tests assert that authorization is enforced by the backend itself and is
never delegated to the language model:

* unpublished / rejected document versions are never retrievable,
* employees cannot access another employee's onboarding data,
* sensitive requests are rejected and escalated.
"""

import httpx
import pytest

from app.core.config import Settings
from app.main import app
from app.models.document import Document, DocumentType, VersionStatus
from app.repositories.firestore import FirestoreRepository, get_firestore_repository
from app.repositories.storage import StorageRepository, get_storage_repository
from app.services.document_processor import DocumentProcessor, get_document_processor
from app.services.embeddings import EmbeddingService
from app.services.retrieval import RetrievalService
from app.services.router import ScopeRouter
from tests.conftest import make_test_pdf


@pytest.fixture
def isolated_repos(tmp_path):
    settings = Settings(local_storage_dir=str(tmp_path / "storage"))
    firestore_repo = FirestoreRepository(settings=settings)
    storage_repo = StorageRepository(settings=settings)
    embedding_service = EmbeddingService(settings=settings)
    processor = DocumentProcessor(
        firestore_repo=firestore_repo,
        storage_repo=storage_repo,
        embedding_service=embedding_service,
        settings=settings,
    )
    return settings, firestore_repo, storage_repo, embedding_service, processor


@pytest.mark.asyncio
async def test_unpublished_version_is_not_retrievable(isolated_repos):
    settings, firestore_repo, _storage, embedding_service, processor = isolated_repos

    doc = Document(
        id="secret-policy",
        title="Secret Policy",
        document_type=DocumentType.POLICY,
        department="People Operations",
        location="Jakarta",
    )
    await firestore_repo.create_document(doc)

    version = await processor.upload_version("secret-policy", make_test_pdf([
        "1. Confidential Rollout\nThis unpublished secret policy describes the confidential rollout plan.",
    ]))
    # Upload == PROCESSING/REVIEW, NOT published.
    await processor.process_version(version.id)

    retrieval = RetrievalService(
        firestore_repo=firestore_repo, embedding_service=embedding_service
    )
    results = await retrieval.search("What is the confidential rollout plan?", top_k=5)
    assert results == [], "unpublished document must never be retrievable"


@pytest.mark.asyncio
async def test_rejected_version_is_not_retrievable(isolated_repos):
    settings, firestore_repo, _storage, embedding_service, processor = isolated_repos

    doc = Document(
        id="rejected-policy",
        title="Rejected Policy",
        document_type=DocumentType.POLICY,
        department="People Operations",
        location="Jakarta",
    )
    await firestore_repo.create_document(doc)

    version = await processor.upload_version("rejected-policy", make_test_pdf([
        "1. Draft Content\nThis content was rejected by HR and must never reach employees.",
    ]))
    await processor.process_version(version.id)

    stored = await firestore_repo.get_document_version(version.id)
    stored.status = VersionStatus.REJECTED
    await firestore_repo.update_document_version(stored)

    retrieval = RetrievalService(
        firestore_repo=firestore_repo, embedding_service=embedding_service
    )
    results = await retrieval.search("What is the rejected draft content?", top_k=5)
    assert results == [], "rejected document must never be retrievable"


@pytest.mark.asyncio
async def test_cross_user_task_access_is_forbidden(isolated_repos):
    _settings, firestore_repo, storage_repo, embedding_service, processor = isolated_repos

    app.dependency_overrides[get_firestore_repository] = lambda: firestore_repo
    app.dependency_overrides[get_storage_repository] = lambda: storage_repo
    app.dependency_overrides[get_document_processor] = lambda: processor
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Seed user_001's tasks.
            r = await client.get("/api/v1/tasks", headers={"X-User-Id": "user_001"})
            tasks = r.json()
            assert tasks

            # user_002 must not be able to complete user_001's task.
            r = await client.post(
                f"/api/v1/tasks/{tasks[0]['id']}/complete",
                headers={"X-User-Id": "user_002"},
            )
            assert r.status_code == 403
            assert r.json()["error"]["code"] == "UNAUTHORIZED"
    finally:
        app.dependency_overrides.clear()


def test_sensitive_requests_are_rejected():
    intent, allowed, reason = ScopeRouter.route("What is the salary of my manager?")
    assert allowed is False
    assert "confidential" in reason.lower() or "cannot" in reason.lower()


def test_out_of_scope_requests_are_rejected():
    intent, allowed, reason = ScopeRouter.route("Tell me a joke about cats")
    assert allowed is False


@pytest.mark.asyncio
async def test_sensitive_chat_request_is_escalated(isolated_repos):
    _settings, firestore_repo, storage_repo, _embedding, processor = isolated_repos

    app.dependency_overrides[get_firestore_repository] = lambda: firestore_repo
    app.dependency_overrides[get_storage_repository] = lambda: storage_repo
    app.dependency_overrides[get_document_processor] = lambda: processor
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/v1/chat",
                headers={"X-User-Id": "user_001"},
                json={"user_id": "user_001", "message": "How much does my colleague earn?"},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["intent"] == "OUT_OF_SCOPE"
            assert body["grounded"] is False
            assert body["escalation_required"] is True
            assert body["citations"] == []
    finally:
        app.dependency_overrides.clear()
