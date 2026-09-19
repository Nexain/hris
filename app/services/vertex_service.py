import logging
import os
from typing import AsyncIterator, List, Optional

from fastapi import HTTPException, status
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError

from app.config import Settings, get_settings
from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse, TokenUsage

logger = logging.getLogger(__name__)


class VertexGeminiService:
    """Service wrapping Vertex AI Gemini model interactions."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[genai.Client] = None,
    ):
        self.settings = settings or get_settings()

        if self.settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                self.settings.google_application_credentials
            )

        self._client = client

    @property
    def client(self) -> genai.Client:
        """Lazy-initialize and return the Google GenAI client for Vertex AI."""
        if self._client is None:
            project_id = self.settings.gcp_project_id.strip()
            if not project_id or project_id == "your-gcp-project-id":
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=(
                        "GCP Project ID is not configured. Please set GCP_PROJECT_ID "
                        "in your .env file or environment variables."
                    ),
                )

            try:
                self._client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=self.settings.gcp_location,
                )
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI client: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Vertex AI initialization error: {str(e)}",
                )
        return self._client

    def _build_contents(
        self, message: str, history: Optional[List[ChatMessage]] = None
    ) -> List[types.Content]:
        """Convert conversational history and current message into Vertex AI Content objects."""
        contents: List[types.Content] = []

        if history:
            for msg in history:
                if msg.role == "system":
                    # System messages are passed via config.system_instruction
                    continue
                role = "user" if msg.role == "user" else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )

        # Append current user prompt
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)],
            )
        )
        return contents

    def _build_config(self, request: ChatRequest) -> types.GenerateContentConfig:
        """Build generation configuration including temperature and system instruction."""
        system_instruction = request.system_instruction

        # If system messages were included in history, also collect them
        if request.history:
            system_messages = [
                msg.content for msg in request.history if msg.role == "system"
            ]
            if system_messages:
                combined = "\n".join(system_messages)
                system_instruction = (
                    f"{combined}\n{system_instruction}"
                    if system_instruction
                    else combined
                )

        config_kwargs = {}
        if request.temperature is not None:
            config_kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = request.max_output_tokens
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        return types.GenerateContentConfig(**config_kwargs)

    async def generate_chat(self, request: ChatRequest) -> ChatResponse:
        """Generate a chat response using Gemini via Vertex AI."""
        target_model = request.model or self.settings.gemini_model
        contents = self._build_contents(request.message, request.history)
        config = self._build_config(request)

        try:
            response = await self.client.aio.models.generate_content(
                model=target_model,
                contents=contents,
                config=config,
            )

            # Extract usage metadata if available
            usage: Optional[TokenUsage] = None
            if response.usage_metadata:
                usage = TokenUsage(
                    prompt_tokens=response.usage_metadata.prompt_token_count,
                    candidates_tokens=response.usage_metadata.candidates_token_count,
                    total_tokens=response.usage_metadata.total_token_count,
                )

            # Extract finish reason if available
            finish_reason: Optional[str] = None
            if response.candidates and len(response.candidates) > 0:
                first_candidate = response.candidates[0]
                if getattr(first_candidate, "finish_reason", None) is not None:
                    finish_reason = str(first_candidate.finish_reason)

            response_text = response.text or ""

            return ChatResponse(
                response=response_text,
                model=target_model,
                usage=usage,
                finish_reason=finish_reason,
            )

        except (APIError, ClientError, ServerError) as e:
            logger.error(f"Vertex AI API error: {e}")
            status_code = getattr(e, "code", status.HTTP_502_BAD_GATEWAY)
            # Ensure valid HTTP status code
            if not isinstance(status_code, int) or status_code < 100 or status_code > 599:
                status_code = status.HTTP_502_BAD_GATEWAY
            raise HTTPException(
                status_code=status_code,
                detail=f"Vertex AI Gemini Error: {e.message if hasattr(e, 'message') else str(e)}",
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error communicating with Vertex AI: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate response from Gemini: {str(e)}",
            )

    async def generate_chat_stream(
        self, request: ChatRequest
    ) -> AsyncIterator[str]:
        """Generate a streaming chat response using Gemini via Vertex AI."""
        target_model = request.model or self.settings.gemini_model
        contents = self._build_contents(request.message, request.history)
        config = self._build_config(request)

        try:
            response_stream = await self.client.aio.models.generate_content_stream(
                model=target_model,
                contents=contents,
                config=config,
            )

            async for chunk in response_stream:
                chunk_text = chunk.text or ""
                if chunk_text:
                    yield chunk_text

        except (APIError, ClientError, ServerError) as e:
            logger.error(f"Vertex AI streaming error: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Vertex AI streaming error: {str(e)}",
            )
        except Exception as e:
            logger.exception(f"Unexpected error in streaming response: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Streaming generation error: {str(e)}",
            )


# Global singleton holder for dependency injection
_vertex_service_instance: Optional[VertexGeminiService] = None


def get_vertex_service() -> VertexGeminiService:
    """Dependency provider for VertexGeminiService."""
    global _vertex_service_instance
    if _vertex_service_instance is None:
        _vertex_service_instance = VertexGeminiService()
    return _vertex_service_instance

