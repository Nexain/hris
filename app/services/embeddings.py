import hashlib
import logging
import math
import re
from typing import List, Optional

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Minimal English stopword set for the deterministic offline embedding fallback.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "should",
    "that", "the", "their", "them", "this", "to", "we", "what", "when", "where",
    "which", "who", "will", "with", "you", "your",
}


def _lexical_embedding(text: str, dim: int = 768) -> List[float]:
    """
    Deterministic bag-of-words embedding using feature hashing.

    Used only as an offline / test fallback when Vertex AI embeddings are not
    configured. Produces L2-normalized vectors so that lexical overlap between a
    query and a chunk yields a meaningful cosine similarity, which lets the RAG
    pipeline and retrieval evaluation run without Google Cloud credentials.
    """
    vec = [0.0] * dim
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]
    grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]

    for gram in grams:
        # md5 (not built-in hash) for stable, process-independent hashing.
        h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


class EmbeddingService:
    """Service generating vector embeddings using Vertex AI or deterministic offline fallback."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._client = None

    @property
    def client(self):
        """Lazy-initialize Google GenAI Client if GCP is configured."""
        if self._client is None:
            project_id = self.settings.gcp_project_id.strip()
            if project_id and project_id != "your-gcp-project-id":
                try:
                    from google import genai
                    self._client = genai.Client(
                        vertexai=True,
                        project=project_id,
                        location=self.settings.gcp_location,
                    )
                except Exception as e:
                    logger.warning(f"Could not initialize GenAI client for embeddings: {e}")
                    self._client = None
        return self._client

    def _generate_mock_embedding(self, text: str, dim: int = 768) -> List[float]:
        """Generate a deterministic, lexical vector for offline/testing purposes."""
        return _lexical_embedding(text, dim)


    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        embeddings = await self.get_embeddings([text])
        return embeddings[0] if embeddings else self._generate_mock_embedding(text)

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for multiple chunks."""
        if not texts:
            return []

        if self.client:
            try:
                # Use client.aio.models.embed_content
                results = []
                for text in texts:
                    res = await self.client.aio.models.embed_content(
                        model=self.settings.embedding_model,
                        contents=text,
                    )
                    if hasattr(res, "embedding") and res.embedding:
                        results.append(res.embedding.values)
                    elif hasattr(res, "embeddings") and res.embeddings:
                        results.append(res.embeddings[0].values)
                    else:
                        results.append(self._generate_mock_embedding(text))
                return results
            except Exception as e:
                logger.warning(f"Embedding generation via Vertex AI failed: {e}. Falling back to mock embeddings.")

        return [self._generate_mock_embedding(t) for t in texts]


_embedding_service_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Dependency provider for EmbeddingService."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance
