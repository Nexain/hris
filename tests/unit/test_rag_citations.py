from app.models.document import DocumentChunk
from app.schemas.chat import ChatMessage
from app.services.gemini import build_retrieval_query, extract_cited_sources, select_cited_chunks


def _chunk(doc_id: str) -> DocumentChunk:
    return DocumentChunk(id=f"{doc_id}_c0", document_id=doc_id, version_id="v1", content="x", chunk_index=0)


def test_extract_cited_sources_strips_markers_and_dedupes():
    answer, cited = extract_cited_sources(
        "You get twelve days [Source 2]. Claims need receipts [Sources 3, 2]. Bad ref [Source 9].", 3)
    assert answer == "You get twelve days. Claims need receipts. Bad ref."
    assert cited == [2, 3]


def test_select_cited_chunks_prefers_model_citations():
    scored = [(_chunk("a"), 0.9), (_chunk("b"), 0.88), (_chunk("c"), 0.5)]
    assert [c.document_id for c, _ in select_cited_chunks(scored, [2])] == ["b"]


def test_select_cited_chunks_falls_back_to_relevance_cutoff():
    scored = [(_chunk("a"), 0.9), (_chunk("b"), 0.8), (_chunk("c"), 0.5)]
    assert [c.document_id for c, _ in select_cited_chunks(scored, [])] == ["a", "b"]


def test_retrieval_query_includes_previous_user_question():
    history = [
        ChatMessage(role="user", content="How many days of annual leave do I get?"),
        ChatMessage(role="model", content="Twelve days."),
    ]
    query = build_retrieval_query("And can I carry it over?", history)
    assert "annual leave" in query and "carry it over" in query
    assert build_retrieval_query("Hi", []) == "Hi"
