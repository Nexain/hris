from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"


class UserProfile(BaseModel):
    user_id: str
    name: str = "Aji"
    role: str = "Backend Engineer"
    department: str = "Engineering"
    location: str = "Jakarta"
    access_level: str = "EMPLOYEE"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserProfileCreate(BaseModel):
    user_id: Optional[str] = None
    name: str
    role: str
    department: str
    location: str


class UserProfileUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None
    location: Optional[str] = None


class OnboardingTask(BaseModel):
    id: str = Field(..., description="Task identifier, e.g. task_001")
    user_id: str
    title: str
    description: str
    day: int = Field(default=1, ge=1, le=30)
    source_document_id: Optional[str] = None
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    completed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TaskCompleteResponse(BaseModel):
    task_id: str
    status: TaskStatus


class OnboardingProgress(BaseModel):
    total: int
    completed: int
    remaining: int
    progress_percent: float
