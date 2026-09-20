from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Represents a single message in a chat conversation."""
    role: Literal["user", "model", "system"] = Field(
        ...,
        description="The role of the entity sending the message."
    )
    content: str = Field(
        ...,
        min_length=1,
        description="The text content of the message."
    )


class ChatRequest(BaseModel):
    """Request payload for the /chat endpoint."""
    message: str = Field(
        ...,
        min_length=1,
        description="User message / prompt to send to the Gemini model."
    )
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional list of previous messages in the conversation."
    )
    system_instruction: Optional[str] = Field(
        default=None,
        description="Optional system prompt / instructions to steer model behavior."
    )
    model: Optional[str] = Field(
        default=None,
        description="Override the default Vertex AI Gemini model (e.g., gemini-1.5-pro, gemini-2.0-flash)."
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Optional sampling temperature between 0.0 and 2.0."
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional maximum number of tokens to generate."
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response as Server-Sent Events (SSE)."
    )


class TokenUsage(BaseModel):
    """Token usage breakdown."""
    prompt_tokens: Optional[int] = None
    candidates_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    """Response payload for the /chat endpoint."""
    response: str = Field(
        ...,
        description="The generated response text from Gemini."
    )
    model: str = Field(
        ...,
        description="The model used to generate the response."
    )
    usage: Optional[TokenUsage] = Field(
        default=None,
        description="Token consumption details."
    )
    finish_reason: Optional[str] = Field(
        default=None,
        description="The reason why generation finished (e.g., STOP, MAX_TOKENS)."
    )

