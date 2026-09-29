import math
from typing import Any, Dict, List, Optional, Tuple

from app.models.chat import Citation
from app.models.document import DocumentChunk
from app.repositories.firestore import FirestoreRepository, get_firestore_repository
from app.services.embeddings import EmbeddingService, get_embedding_service


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class RetrievalService:
    """Service retrieving relevant knowledge chunks using vector similarity."""

    def __init__(
        self,
        firestore_repo: Optional[FirestoreRepository] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.firestore_repo = firestore_repo or get_firestore_repository()
        self.embedding_service = embedding_service or get_embedding_service()

    async def search(
        self,
        query: str,
        top_k: int = 5,
        department: Optional[str] = None,
        location: Optional[str] = None,
        min_similarity: float = 0.25,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve top-K chunks from published document versions matching query."""
        # 1. Generate query embedding
        query_vector = await self.embedding_service.get_embedding(query)

        # 2. Get published chunks
        published_chunks = await self.firestore_repo.get_published_chunks(
            department=department,
            location=location,
        )

        if not published_chunks:
            return []

        # 3. Score chunks
        scored_chunks: List[Tuple[DocumentChunk, float]] = []
        for chunk in published_chunks:
            if not chunk.embedding:
                continue
            sim = cosine_similarity(query_vector, chunk.embedding)
            if sim >= min_similarity:
                scored_chunks.append((chunk, sim))

        # 4. Sort descending by similarity
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return scored_chunks[:top_k]

    async def search_chunks(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: int = 5,
        min_similarity: float = -1.0,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Rank a caller-provided chunk set (e.g. a not-yet-published version).

        Ranks by similarity rather than filtering, so evaluation sees a full top-K
        even when lexical similarity is negative.
        """
        if not chunks:
            return []

        query_vector = await self.embedding_service.get_embedding(query)
        scored: List[Tuple[DocumentChunk, float]] = []
        for chunk in chunks:
            if not chunk.embedding:
                continue
            sim = cosine_similarity(query_vector, chunk.embedding)
            if sim >= min_similarity:
                scored.append((chunk, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class ContextBuilder:
    """Formats retrieved document chunks into clean LLM context."""

    @staticmethod
    def build_context(scored_chunks: List[Tuple[DocumentChunk, float]]) -> str:
        """Construct structured context text from retrieved chunks."""
        if not scored_chunks:
            return ""

        context_parts = []
        for i, (chunk, score) in enumerate(scored_chunks, 1):
            title = chunk.metadata.get("document_title", chunk.document_id)
            version = chunk.metadata.get("version", "1.0")
            section = chunk.section or "General"
            page_info = f"Page: {chunk.page}" if chunk.page is not None else "Page: N/A"

            context_parts.append(
                f"[Source {i}]\n"
                f"Document: {title}\n"
                f"Document ID: {chunk.document_id}\n"
                f"Version: {version}\n"
                f"Section: {section}\n"
                f"{page_info}\n"
                f"Content:\n{chunk.content}\n"
            )

        return "\n---\n".join(context_parts)

    @staticmethod
    def extract_citations(scored_chunks: List[Tuple[DocumentChunk, float]]) -> List[Citation]:
        """Convert retrieved chunks into structured citation objects."""
        citations = []
        seen = set()

        for chunk, score in scored_chunks:
            key = (chunk.document_id, chunk.metadata.get("version", "1.0"), chunk.section, chunk.page)
            if key in seen:
                continue
            seen.add(key)

            citations.append(
                Citation(
                    document_id=chunk.document_id,
                    document_title=chunk.metadata.get("document_title", chunk.document_id),
                    version=chunk.metadata.get("version", "1.0"),
                    section=chunk.section or "General",
                    page=chunk.page,
                )
            )

        return citations


_retrieval_service_instance: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    """Dependency provider for RetrievalService."""
    global _retrieval_service_instance
    if _retrieval_service_instance is None:
        _retrieval_service_instance = RetrievalService()
    return _retrieval_service_instance
