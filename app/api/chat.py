import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import Settings, get_settings
from app.core.errors import ResourceNotFoundException, UnauthorizedAccessException
from app.core.security import UserContext, get_current_user
from app.models.chat import ChatRequestV1, ChatResponseV1, IntentType
from app.models.onboarding import TaskStatus
from app.schemas.chat import ChatMessage
from app.services.gemini import FALLBACK_MESSAGE, RAGEngine, get_rag_engine
from app.services.intent import IntentClassifier, get_intent_classifier
from app.services.language import detect_language, onboarding_status_answer
from app.services.language import text as localized
from app.services.router import ScopeRouter
from app.services.onboarding import OnboardingService, get_onboarding_service
from app.services.vertex_service import VertexGeminiService, get_vertex_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])

_REFUSAL_KEYS = {
    ScopeRouter.SENSITIVE_REASON: "sensitive",
    ScopeRouter.OUT_OF_SCOPE_REASON: "out_of_scope",
}


def _localize_refusal(reason: str, lang: str) -> str:
    key = _REFUSAL_KEYS.get(reason)
    return localized(lang, key) if key else reason

# Fields that ask for the raw, ungrounded Gemini bridge instead of grounded chat.
BRIDGE_FIELDS = ("stream", "system_instruction", "model", "temperature", "max_output_tokens")

# Roles the frontend may use for earlier turns, mapped to the model's roles.
_HISTORY_ROLES = {"user": "user", "employee": "user", "model": "model", "assistant": "model", "copilot": "model", "bot": "model"}


def _parse_history(raw: Any) -> List[ChatMessage]:
    """Leniently read earlier chat turns from the request; unknown entries are skipped."""
    if not isinstance(raw, list):
        return []
    history: List[ChatMessage] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = _HISTORY_ROLES.get(str(item.get("role") or item.get("sender") or "").lower())
        content = item.get("content")
        if role and isinstance(content, str) and content.strip():
            history.append(ChatMessage(role=role, content=content))
    return history


@router.post(
    "/chat",
    response_model=Union[ChatResponseV1, Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="DayOne AI Onboarding Chat API",
    description="Main chat interface for new employees. Routes intents, queries published company knowledge with vector search and citations, or provides onboarding status.",
)
async def chat_endpoint(
    raw_request: Request,
    current_user: UserContext = Depends(get_current_user),
    rag_engine: RAGEngine = Depends(get_rag_engine),
    onboarding_service: OnboardingService = Depends(get_onboarding_service),
    vertex_service: VertexGeminiService = Depends(get_vertex_service),
    intent_classifier: IntentClassifier = Depends(get_intent_classifier),
    settings: Settings = Depends(get_settings),
):
    """Chat endpoint fulfilling Section 3.2 contract while supporting streaming."""
    body = await raw_request.json()
    message = body.get("message", "").strip()

    if not message:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Field 'message' cannot be empty."},
        )

    # Raw Gemini bridge: free-form, ungrounded output. It is off unless the server sets
    # ENABLE_RAW_GEMINI_BRIDGE, and even then only HR admins may use it, so callers
    # cannot spend Gemini quota on answers that ignore company documents.
    # "history" is not a bridge field: grounded chat accepts it.
    if any(field in body for field in BRIDGE_FIELDS):
        if not (settings.enable_raw_gemini_bridge and current_user.is_hr_admin()):
            raise UnauthorizedAccessException(
                "Chat only answers from company documents. "
                "Custom instructions, models, sampling and streaming are not allowed."
            )
        from app.schemas.chat import ChatRequest
        stream_req = ChatRequest(**{k: v for k, v in body.items() if k != "user_id"})

        if body.get("stream", False):
            async def event_generator():
                try:
                    async for chunk in vertex_service.generate_chat_stream(stream_req):
                        data = json.dumps({"text": chunk})
                        yield f"data: {data}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    err_data = json.dumps({"error": str(e)})
                    yield f"data: {err_data}\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        result = await vertex_service.generate_chat(stream_req)
        # The legacy bridge response model is not part of this route's response_model,
        # so serialize it directly to bypass response validation.
        if hasattr(result, "model_dump"):
            return JSONResponse(content=result.model_dump(mode="json"))
        return JSONResponse(content=result)

    # The caller's identity comes from the request headers. A different user_id in the
    # body is refused, so one employee cannot read another's onboarding data.
    user_id = current_user.user_id
    body_user_id = body.get("user_id")
    if body_user_id and body_user_id != user_id:
        raise UnauthorizedAccessException("You can only chat as yourself.")

    history = _parse_history(body.get("history"))

    lang = detect_language(message)

    # Users without an onboarding profile (e.g. HR admins testing policies) can still
    # ask company questions; fall back to the department/location from the request.
    async def search_documents():
        try:
            profile = await onboarding_service.get_profile(user_id)
            department, location = profile.department, profile.location
        except ResourceNotFoundException:
            department, location = current_user.department, current_user.location
        return department, location, await rag_engine.retrieve(message, history, department, location)

    # 1. Scope Gate & Intent Routing (Phase 5).
    # Regex rules decide first with no network call. Otherwise the Gemini intent call
    # and the document search run at the same time, and the search result is used if
    # the message turns out to be a company-knowledge question.
    search_task: Optional[asyncio.Task] = None
    routed = intent_classifier.fast_route(message)
    if routed is None:
        classify_task = asyncio.create_task(intent_classifier.classify_slow(message, history))
        await asyncio.sleep(0)  # let the Gemini request go out before the search starts
        search_task = asyncio.create_task(search_documents())
        # If the search is not used, still collect its exception so asyncio doesn't log it.
        search_task.add_done_callback(lambda t: t.cancelled() or t.exception())
        try:
            routed = await classify_task
        except BaseException:
            search_task.cancel()
            raise
    intent, is_allowed, fallback_reason = routed

    if intent != IntentType.COMPANY_KNOWLEDGE and search_task is not None:
        search_task.cancel()

    if not is_allowed:
        return ChatResponseV1(
            answer=_localize_refusal(fallback_reason, lang),
            citations=[],
            intent=intent,
            grounded=False,
            escalation_required=True,
        )

    # 2. Handle onboarding intents (Phase 6)
    if intent in (
        IntentType.MY_ONBOARDING,
        IntentType.ONBOARDING_ACTION,
        IntentType.GENERAL_ONBOARDING,
    ):
        try:
            tasks = await onboarding_service.get_tasks(user_id)
            progress = await onboarding_service.get_progress(user_id)
        except ResourceNotFoundException:
            tasks, progress = [], None
        pending = [t for t in tasks if t.status == TaskStatus.PENDING]

        if progress is None:
            answer = localized(lang, "no_profile")
        else:
            answer = onboarding_status_answer(lang, tasks, pending, progress)

        if intent == IntentType.GENERAL_ONBOARDING:
            answer = localized(lang, "general_intro") + answer

        return ChatResponseV1(
            answer=answer,
            citations=[],
            intent=intent,
            grounded=True,
            escalation_required=False,
        )

    # 3. Handle COMPANY_KNOWLEDGE via RAG (Phase 3)
    if search_task is None:
        search_task = asyncio.create_task(search_documents())
    department, location, scored_chunks = await search_task

    answer, citations, grounded, escalation_required = await rag_engine.answer_question(
        question=message,
        user_id=user_id,
        department=department,
        location=location,
        history=history,
        scored_chunks=scored_chunks,
    )
    if answer == FALLBACK_MESSAGE:
        answer = localized(lang, "fallback")

    return ChatResponseV1(
        answer=answer,
        citations=citations,
        intent=IntentType.COMPANY_KNOWLEDGE,
        grounded=grounded,
        escalation_required=escalation_required,
    )
