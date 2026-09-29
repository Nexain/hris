import logging
from typing import List, Optional, Tuple

from app.core.config import Settings, get_settings
from app.models.chat import Citation, IntentType
from app.models.document import DocumentChunk
from app.services.retrieval import ContextBuilder, RetrievalService, get_retrieval_service
from app.services.vertex_service import VertexGeminiService, get_vertex_service

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    "I couldn't find reliable information about that in the available company documentation. "
    "Please contact People Operations for assistance."
)

SYSTEM_PROMPT = """You are DayOne AI, a professional and helpful onboarding copilot for new employees.
Your objective is to answer the employee's question accurately using ONLY the company context provided below.

Strict Guidelines:
1. Ground your answer strictly in the provided company document excerpts.
2. If the context does not contain sufficient or reliable evidence to answer the question, respond with:
"I couldn't find reliable information about that in the available company documentation. Please contact People Operations for assistance."
3. Do not invent company policies, benefits, dates, or contact details.
4. Keep the response helpful, clear, and professional.
"""


class RAGEngine:
    """Coordinates retrieval, context assembly, and Gemini generation for company questions."""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        vertex_service: Optional[VertexGeminiService] = None,
        settings: Optional[Settings] = None,
    ):
        self.retrieval_service = retrieval_service or get_retrieval_service()
        self.vertex_service = vertex_service or get_vertex_service()
        self.settings = settings or get_settings()

    async def answer_question(
        self,
        question: str,
        user_id: str = "user_001",
        department: Optional[str] = None,
        location: Optional[str] = None,
    ) -> Tuple[str, List[Citation], bool, bool]:
        """
        Execute full RAG pipeline for a company knowledge question.
        Returns: (answer, citations, grounded, escalation_required)
        """
        # 1. Retrieve top-K chunks from published versions
        scored_chunks = await self.retrieval_service.search(
            query=question,
            top_k=5,
            department=department,
            location=location,
        )

        # 2. Safe Fallback if no relevant published context found
        if not scored_chunks:
            return FALLBACK_MESSAGE, [], False, True

        # 3. Build context
        context_text = ContextBuilder.build_context(scored_chunks)
        citations = ContextBuilder.extract_citations(scored_chunks)

        # 4. Generate response using Gemini
        prompt = (
            f"Employee Question: {question}\n\n"
            f"Company Document Context:\n{context_text}\n\n"
            f"Please answer the employee's question based strictly on the above context."
        )

        try:
            from app.schemas.chat import ChatRequest
            req = ChatRequest(
                message=prompt,
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
            )
            llm_response = await self.vertex_service.generate_chat(req)
            answer_text = llm_response.response.strip()

            # Check if model triggered fallback
            if "couldn't find reliable information" in answer_text.lower() or "not enough information" in answer_text.lower():
                return FALLBACK_MESSAGE, [], False, True

            return answer_text, citations, True, False

        except Exception as e:
            logger.warning(f"Gemini call in RAG pipeline failed: {e}. Returning context summary.")
            # If offline / mock mode where Vertex client is not active, synthesize grounded answer from top chunk
            top_chunk = scored_chunks[0][0]
            summary_answer = f"Based on {top_chunk.metadata.get('document_title', 'company documentation')}: {top_chunk.content}"
            return summary_answer, citations, True, False


_rag_engine_instance: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Dependency provider for RAGEngine."""
    global _rag_engine_instance
    if _rag_engine_instance is None:
        _rag_engine_instance = RAGEngine()
    return _rag_engine_instance
