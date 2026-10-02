"""Retrieval, answer and citation evaluation for the DayOne AI knowledge base.

Implements the evaluation deliverables from spec §16:

* Golden-question retrieval evaluation (Hit@K, MRR).
* Citation validation against retrieved chunks.
* A structured, reusable result object for regression tracking across document
  versions.

The evaluator is intentionally infrastructure-light: it only depends on the
:class:`RetrievalService`, so it can run offline with the deterministic
embedding fallback or against real Vertex AI embeddings in production.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from app.models.chat import Citation
from app.models.document import DocumentChunk
from app.services.retrieval import RetrievalService, get_retrieval_service


@dataclass
class GoldenQuestion:
    """A persistent regression question (spec §6.5 `evaluation_questions`)."""

    question: str
    expected_document_id: str
    expected_answer: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class QuestionResult:
    """Per-question retrieval outcome."""

    question: str
    expected_document_id: str
    retrieved_document_ids: List[str]
    hit: bool
    rank: Optional[int] = None


@dataclass
class RetrievalEvaluation:
    """Aggregate retrieval metrics over a golden question set."""

    total: int
    hits: int
    hit_rate: float
    mrr: float
    top_k: int
    results: List[QuestionResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Convenience flag: whether the spec's initial Hit@5 >= 80% target is met."""
        return self.hit_rate >= 0.8


class RetrievalEvaluator:
    """Computes Hit@K and MRR for a golden question set against the published index."""

    def __init__(self, retrieval_service: Optional[RetrievalService] = None):
        self.retrieval_service = retrieval_service or get_retrieval_service()

    async def evaluate(
        self,
        golden_questions: Sequence[GoldenQuestion],
        top_k: int = 5,
        department: Optional[str] = None,
        location: Optional[str] = None,
    ) -> RetrievalEvaluation:
        """Run each golden question through retrieval and compute Hit@K and MRR."""
        results: List[QuestionResult] = []
        hits = 0
        reciprocal_rank_sum = 0.0

        for gq in golden_questions:
            scored = await self.retrieval_service.search(
                query=gq.question,
                top_k=top_k,
                department=department,
                location=location,
            )
            retrieved_ids = [chunk.document_id for chunk, _score in scored]

            rank: Optional[int] = None
            for idx, doc_id in enumerate(retrieved_ids):
                if doc_id == gq.expected_document_id:
                    rank = idx + 1
                    break

            hit = rank is not None
            if hit:
                hits += 1
                reciprocal_rank_sum += 1.0 / rank

            results.append(
                QuestionResult(
                    question=gq.question,
                    expected_document_id=gq.expected_document_id,
                    retrieved_document_ids=retrieved_ids,
                    hit=hit,
                    rank=rank,
                )
            )

        total = len(golden_questions)
        hit_rate = (hits / total) if total else 0.0
        mrr = (reciprocal_rank_sum / total) if total else 0.0

        return RetrievalEvaluation(
            total=total,
            hits=hits,
            hit_rate=round(hit_rate, 4),
            mrr=round(mrr, 4),
            top_k=top_k,
            results=results,
        )

    async def evaluate_version(
        self,
        document_id: str,
        version_id: str,
        firestore_repo,
        top_k: int = 5,
    ) -> RetrievalEvaluation:
        """Run persisted golden questions against a specific version's chunks.

        Used for regression testing across document versions (spec §16.4): the
        version's own chunks are scored directly, so an incoming (REVIEW) version
        can be validated before it is published.
        """
        stored = await firestore_repo.get_evaluation_questions(document_id)
        golden = [
            GoldenQuestion(
                question=q.question,
                expected_document_id=q.document_id,
                expected_answer=q.expected_answer,
                notes=q.expected_source,
            )
            for q in stored
        ]

        chunks = await firestore_repo.get_chunks_by_version(version_id)

        results: List[QuestionResult] = []
        hits = 0
        reciprocal_rank_sum = 0.0

        for gq in golden:
            scored = await self.retrieval_service.search_chunks(gq.question, chunks, top_k=top_k)
            retrieved_ids = [chunk.document_id for chunk, _score in scored]

            rank: Optional[int] = None
            for idx, doc_id in enumerate(retrieved_ids):
                if doc_id == gq.expected_document_id:
                    rank = idx + 1
                    break

            hit = rank is not None
            if hit:
                hits += 1
                reciprocal_rank_sum += 1.0 / rank

            results.append(
                QuestionResult(
                    question=gq.question,
                    expected_document_id=gq.expected_document_id,
                    retrieved_document_ids=retrieved_ids,
                    hit=hit,
                    rank=rank,
                )
            )

        total = len(golden)
        hit_rate = (hits / total) if total else 0.0
        mrr = (reciprocal_rank_sum / total) if total else 0.0

        return RetrievalEvaluation(
            total=total,
            hits=hits,
            hit_rate=round(hit_rate, 4),
            mrr=round(mrr, 4),
            top_k=top_k,
            results=results,
        )


def validate_citations(
    citations: Sequence[Citation],
    retrieved_chunks: Sequence[Tuple[DocumentChunk, float]],
) -> bool:
    """
    Validate that every returned citation points to a retrieved chunk (spec §16.2).

    A citation is considered valid when its (document_id, version, section, page)
    matches at least one retrieved chunk. Returns ``True`` only when there is at
    least one citation and all of them are grounded in retrieved context.
    """
    if not citations:
        return False

    retrieved_keys = {
        (chunk.document_id, chunk.metadata.get("version", "1.0"), chunk.section, chunk.page)
        for chunk, _score in retrieved_chunks
    }

    for citation in citations:
        key = (citation.document_id, citation.version, citation.section, citation.page)
        if key not in retrieved_keys:
            return False
    return True
