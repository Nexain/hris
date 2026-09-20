import json
from typing import Union
from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.vertex_service import VertexGeminiService, get_vertex_service

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=Union[ChatResponse, None],
    responses={
        status.HTTP_200_OK: {
            "description": "Successful response or event-stream when stream=True",
            "content": {
                "application/json": {
                    "schema": ChatResponse.model_json_schema()
                },
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "example": 'data: {"text": "Hello! How can I help you today?"}\\n\\ndata: [DONE]\\n\\n',
                    }
                },
            },
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Configuration or internal server error"
        },
        status.HTTP_502_BAD_GATEWAY: {
            "description": "Error returned from Google Cloud Vertex AI"
        },
    },
    summary="Chat with Gemini on Vertex AI",
    description="Send a message or conversation history to Gemini on Google Cloud Vertex AI and receive a response. Supports standard JSON responses as well as Server-Sent Events (SSE) streaming.",
)
async def chat_endpoint(
    request: ChatRequest,
    service: VertexGeminiService = Depends(get_vertex_service),
):
    """Bridge endpoint to communicate with Gemini in Vertex AI."""
    if request.stream:
        async def event_generator():
            try:
                async for chunk in service.generate_chat_stream(request):
                    data = json.dumps({"text": chunk})
                    yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                error_payload = json.dumps({"error": str(e)})
                yield f"data: {error_payload}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return await service.generate_chat(request)

