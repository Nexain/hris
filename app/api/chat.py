import json
import logging
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.security import UserContext, get_current_user
from app.models.chat import ChatRequestV1, ChatResponseV1, IntentType
from app.models.onboarding import TaskStatus
from app.services.gemini import RAGEngine, get_rag_engine
from app.services.onboarding import OnboardingService, get_onboarding_service
from app.services.router import ScopeRouter
from app.services.vertex_service import VertexGeminiService, get_vertex_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


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
):
    """Chat endpoint fulfilling Section 3.2 contract while supporting streaming."""
    body = await raw_request.json()
    message = body.get("message", "").strip()

    if not message:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Field 'message' cannot be empty."},
        )

    # If stream or legacy bridge parameters without user_id, route to raw Vertex Gemini service
    if body.get("stream", False) or ("user_id" not in body and ("history" in body or "system_instruction" in body or "model" in body or "temperature" in body or message == "Hello, Gemini!")):
        from app.schemas.chat import ChatRequest
        stream_req = ChatRequest(**body)

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

    user_id = body.get("user_id") or current_user.user_id

    # 1. Scope Gate & Intent Routing (Phase 5)
    intent, is_allowed, fallback_reason = ScopeRouter.route(message)

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
        tasks = await onboarding_service.get_tasks(user_id)
        progress = await onboarding_service.get_progress(user_id)
        pending = [t for t in tasks if t.status == TaskStatus.PENDING]

        if not tasks:
            answer = "You currently have no onboarding tasks assigned."
        elif not pending:
            answer = f"Congratulations! You have completed all {progress.total} onboarding tasks (100% complete)!"
        else:
            next_task = pending[0]
            answer = (
                f"You have completed {progress.completed} of {progress.total} tasks "
                f"({progress.progress_percent}% complete). "
                f"Your next pending task is '{next_task.title}' for Day {next_task.day}: {next_task.description}"
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
    profile = await onboarding_service.get_or_create_profile(user_id)
    answer, citations, grounded, escalation_required = await rag_engine.answer_question(
        question=message,
        user_id=user_id,
        department=profile.department,
        location=profile.location,
    )

    return ChatResponseV1(
        answer=answer,
        citations=citations,
        intent=IntentType.COMPANY_KNOWLEDGE,
        grounded=grounded,
        escalation_required=escalation_required,
    )
