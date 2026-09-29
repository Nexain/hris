from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IntentType(str, Enum):
    COMPANY_KNOWLEDGE = "COMPANY_KNOWLEDGE"
    MY_ONBOARDING = "MY_ONBOARDING"
    ONBOARDING_ACTION = "ONBOARDING_ACTION"
    GENERAL_ONBOARDING = "GENERAL_ONBOARDING"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class Citation(BaseModel):
    document_id: str
    document_title: str
    version: str = "1.0"
    section: Optional[str] = "General"
    page: Optional[int] = None


class ChatRequestV1(BaseModel):
    user_id: str = Field(default="user_001", description="Employee user ID")
    message: str = Field(..., min_length=1, description="Question or prompt")


class ChatResponseV1(BaseModel):
    answer: str = Field(..., description="Answer to the employee question")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    intent: IntentType = Field(default=IntentType.COMPANY_KNOWLEDGE, description="Detected intent")
    grounded: bool = Field(default=True, description="Whether the answer is grounded in retrieved documents")
    escalation_required: bool = Field(default=False, description="Whether HR escalation is required")
