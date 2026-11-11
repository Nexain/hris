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
from app.services.gemini import RAGEngine, get_rag_engine
from app.services.intent import IntentClassifier, get_intent_classifier
from app.services.onboarding import OnboardingService, get_onboarding_service
from app.services.vertex_service import VertexGeminiService, get_vertex_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])

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

    # 1. Scope Gate & Intent Routing (Phase 5): sensitive regex, then Gemini, then regex fallback.
    intent, is_allowed, fallback_reason = await intent_classifier.classify(message, history)

    if not is_allowed:
        return ChatResponseV1(
            answer=fallback_reason,
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
            answer = "You don't have an onboarding profile yet, so there are no onboarding tasks to show."
        elif not tasks:
            answer = "You currently have no onboarding tasks assigned."
        elif not pending:
            answer = f"Congratulations! You have completed all {progress.total} onboarding tasks (100% complete)!"
        else:
            pending.sort(key=lambda t: (t.day, t.id))
            next_task = pending[0]
            task_lines = "\n".join(f"- Day {t.day}: {t.title}" for t in pending)
            answer = (
                f"You have completed {progress.completed} of {progress.total} tasks "
                f"({progress.progress_percent}% complete). "
                f"You still have {len(pending)} task{'s' if len(pending) != 1 else ''} left:\n\n"
                f"{task_lines}\n\n"
                f"Next up: **{next_task.title}** (Day {next_task.day}). {next_task.description}"
            )

        if intent == IntentType.GENERAL_ONBOARDING:
            answer = (
                "Your onboarding program walks you through company policies and your first-week tasks. "
                + answer
            )

        return ChatResponseV1(
            answer=answer,
            citations=[],
            intent=intent,
            grounded=True,
            escalation_required=False,
        )

    # 3. Handle COMPANY_KNOWLEDGE via RAG (Phase 3)
    # Users without an onboarding profile (e.g. HR admins testing policies) can still
    # ask company questions; fall back to the department/location from the request.
    try:
        profile = await onboarding_service.get_profile(user_id)
        department, location = profile.department, profile.location
    except ResourceNotFoundException:
        department, location = current_user.department, current_user.location

    answer, citations, grounded, escalation_required = await rag_engine.answer_question(
        question=message,
        user_id=user_id,
        department=department,
        location=location,
        history=history,
    )

    return ChatResponseV1(
        answer=answer,
        citations=citations,
        intent=IntentType.COMPANY_KNOWLEDGE,
        grounded=grounded,
        escalation_required=escalation_required,
    )
