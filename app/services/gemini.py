import logging
import re
from typing import List, Optional, Sequence, Tuple

from app.core.config import Settings, get_settings
from app.models.chat import Citation, IntentType
from app.models.document import DocumentChunk
from app.schemas.chat import ChatMessage
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
5. After each statement taken from the context, cite its source as [Source N], using the
numbers of the excerpts below. Cite only the sources you actually used.
6. Earlier conversation turns are for understanding follow-up questions only; facts must still
come from the excerpts.
"""

# Sources scoring below this share of the best match are dropped when the model does
# not cite anything, so loosely related documents are not shown as citations.
RELATIVE_SCORE_CUTOFF = 0.85
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 1500
SOURCE_MARKER = re.compile(r"\[\s*Sources?\s+([\d,\s]+)\]", re.IGNORECASE)


def _trim_history(history: Optional[Sequence[ChatMessage]]) -> List[ChatMessage]:
    """Keep the last few non-system turns, each capped in length."""
    if not history:
        return []
    turns = [m for m in history if m.role != "system" and m.content.strip()]
    return [
        ChatMessage(role=m.role, content=m.content.strip()[:MAX_HISTORY_CHARS])
        for m in turns[-MAX_HISTORY_MESSAGES:]
    ]


def build_retrieval_query(question: str, history: Sequence[ChatMessage]) -> str:
    """Add the previous user question so follow-ups ("and can I carry it over?") find the right documents."""
    previous_user = [m.content for m in history if m.role == "user"]
    if not previous_user:
        return question
    return f"{previous_user[-1]}\n{question}"


def extract_cited_sources(answer: str, source_count: int) -> Tuple[str, List[int]]:
    """Return the answer without [Source N] markers and the 1-based source numbers it cited, in order."""
    cited: List[int] = []
    for match in SOURCE_MARKER.finditer(answer):
        for num in re.findall(r"\d+", match.group(1)):
            n = int(num)
            if 1 <= n <= source_count and n not in cited:
                cited.append(n)
    cleaned = SOURCE_MARKER.sub("", answer)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip(), cited


def select_cited_chunks(
    scored_chunks: List[Tuple[DocumentChunk, float]], cited: List[int]
) -> List[Tuple[DocumentChunk, float]]:
    """Pick the chunks to show as citations: those the model cited, else the clearly relevant ones."""
    if cited:
        return [scored_chunks[n - 1] for n in cited]
    if not scored_chunks:
        return []
    best = scored_chunks[0][1]
    return [(c, s) for c, s in scored_chunks if s >= best * RELATIVE_SCORE_CUTOFF]


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
        history: Optional[Sequence[ChatMessage]] = None,
    ) -> Tuple[str, List[Citation], bool, bool]:
        """
        Execute full RAG pipeline for a company knowledge question.
        Returns: (answer, citations, grounded, escalation_required)
        """
        history = _trim_history(history)

        # 1. Retrieve top-K chunks from published versions
        scored_chunks = await self.retrieval_service.search(
            query=build_retrieval_query(question, history),
            top_k=5,
            department=department,
            location=location,
        )

        # 2. Safe Fallback if no relevant published context found
        if not scored_chunks:
            return FALLBACK_MESSAGE, [], False, True

        # 3. Build context
        context_text = ContextBuilder.build_context(scored_chunks)

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
                history=history or None,
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
            )
            llm_response = await self.vertex_service.generate_chat(req)
            answer_text = llm_response.response.strip()

            # Check if model triggered fallback
            if "couldn't find reliable information" in answer_text.lower() or "not enough information" in answer_text.lower():
                return FALLBACK_MESSAGE, [], False, True

            answer_text, cited = extract_cited_sources(answer_text, len(scored_chunks))
            citations = ContextBuilder.extract_citations(select_cited_chunks(scored_chunks, cited))
            return answer_text, citations, True, False

        except Exception as e:
            logger.warning(f"Gemini call in RAG pipeline failed: {e}. Returning context summary.")
            # If offline / mock mode where Vertex client is not active, synthesize grounded answer from top chunk
            top_chunk = scored_chunks[0][0]
            summary_answer = f"Based on {top_chunk.metadata.get('document_title', 'company documentation')}: {top_chunk.content}"
            # The summary only uses the top chunk, so only that chunk is cited.
            return summary_answer, ContextBuilder.extract_citations(scored_chunks[:1]), True, False


_rag_engine_instance: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Dependency provider for RAGEngine."""
    global _rag_engine_instance
    if _rag_engine_instance is None:
        _rag_engine_instance = RAGEngine()
    return _rag_engine_instance
