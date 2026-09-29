"""Retrieval and citation evaluation tests (spec §16)."""

import pytest
import pytest_asyncio

from app.core.config import Settings
from app.models.document import Document, DocumentType, VersionStatus
from app.repositories.firestore import FirestoreRepository
from app.repositories.storage import StorageRepository
from app.services.document_processor import DocumentProcessor
from app.services.embeddings import EmbeddingService
from app.services.evaluation import (
    GoldenQuestion,
    RetrievalEvaluator,
    validate_citations,
)
from app.services.retrieval import ContextBuilder, RetrievalService
from tests.conftest import make_test_pdf


CORPUS = {
    "laptop-request-procedure": (
        "Laptop Request Procedure",
        "1. Request Process\nTo request a laptop, submit a hardware request form through the IT portal.",
    ),
    "leave-policy": (
        "Leave Policy",
        "1. Annual Leave\nEmployees receive twelve days of annual leave each calendar year.",
    ),
    "it-security-guide": (
        "IT Security Guide",
        "1. Two Factor Authentication\nEnable two factor authentication and use a strong unique password.",
    ),
    "expense-policy": (
        "Expense Policy",
        "1. Reimbursement\nSubmit receipts for reimbursement of expenses within thirty days of purchase.",
    ),
    "remote-work-policy": (
        "Remote Work Policy",
        "1. Eligibility\nEmployees may work remotely up to three days per week with manager approval.",
    ),
    "payroll-calendar": (
        "Payroll Calendar",
        "1. Pay Dates\nSalaries are paid monthly on the twenty fifth business day.",
    ),
}

GOLDEN_QUESTIONS = [
    GoldenQuestion("How do I request a laptop?", "laptop-request-procedure"),
    GoldenQuestion("How many days of annual leave do employees receive?", "leave-policy"),
    GoldenQuestion("Do I need to enable two factor authentication?", "it-security-guide"),
    GoldenQuestion("How do I submit a reimbursement for expenses?", "expense-policy"),
    GoldenQuestion("Can I work remotely during the week?", "remote-work-policy"),
    GoldenQuestion("When are salaries paid?", "payroll-calendar"),
]


@pytest_asyncio.fixture
async def seeded_index(tmp_path):
    """Build an isolated, fully-published knowledge index with a shared repo."""
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

    for doc_id, (title, page_text) in CORPUS.items():
        doc = Document(
            id=doc_id,
            title=title,
            document_type=DocumentType.POLICY,
            department="People Operations",
            location="Jakarta",
        )
        await firestore_repo.create_document(doc)

        pdf = make_test_pdf([page_text])
        version = await processor.upload_version(doc_id, pdf)
        await processor.process_version(version.id)

        version = await firestore_repo.get_document_version(version.id)
        version.status = VersionStatus.PUBLISHED
        await firestore_repo.update_document_version(version)

    retrieval = RetrievalService(
        firestore_repo=firestore_repo,
        embedding_service=embedding_service,
    )
    return retrieval, firestore_repo


@pytest.mark.asyncio
async def test_retrieval_hit_at_5_meets_target(seeded_index):
    """Spec §16.1: Hit@5 >= 80% on the golden evaluation set."""
    retrieval, _ = seeded_index
    evaluator = RetrievalEvaluator(retrieval_service=retrieval)

    result = await evaluator.evaluate(GOLDEN_QUESTIONS, top_k=5)

    assert result.total == len(GOLDEN_QUESTIONS)
    assert result.hit_rate >= 0.8, f"Hit@5 too low: {result.hit_rate} ({result.results})"
    assert result.passed is True
    assert result.mrr > 0.0


@pytest.mark.asyncio
async def test_citations_are_grounded_in_retrieved_chunks(seeded_index):
    """Spec §16.2: citations must point to the exact retrieved source."""
    retrieval, _ = seeded_index

    scored = await retrieval.search("How do I request a laptop?", top_k=5)
    assert scored, "expected at least one retrieved chunk"

    citations = ContextBuilder.extract_citations(scored)
    assert citations
    assert validate_citations(citations, scored) is True

    # A citation pointing at a non-retrieved source must fail validation.
    bogus = ContextBuilder.extract_citations(scored)
    bogus[0].document_id = "not-in-context"
    assert validate_citations(bogus, scored) is False

    # Empty citations must fail validation.
    assert validate_citations([], scored) is False
