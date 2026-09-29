from typing import List
from fastapi import APIRouter, Depends, status

from app.core.security import UserContext, get_current_user
from app.models.onboarding import OnboardingTask, TaskCompleteResponse
from app.services.onboarding import OnboardingService, get_onboarding_service

router = APIRouter(tags=["Tasks"])


@router.get(
    "/tasks",
    response_model=List[OnboardingTask],
    summary="Get onboarding tasks",
    description="Returns the personalized onboarding tasks for the current employee.",
)
async def get_tasks(
    current_user: UserContext = Depends(get_current_user),
    service: OnboardingService = Depends(get_onboarding_service),
):
    """Retrieve personalized onboarding checklist for employee."""
    return await service.get_tasks(current_user.user_id)


@router.post(
    "/tasks/{task_id}/complete",
    response_model=TaskCompleteResponse,
    summary="Complete onboarding task",
    description="Marks a specific onboarding task as COMPLETED.",
)
async def complete_task(
    task_id: str,
    current_user: UserContext = Depends(get_current_user),
    service: OnboardingService = Depends(get_onboarding_service),
):
    """Mark employee task as completed."""
    completed = await service.complete_task(current_user.user_id, task_id)
    return TaskCompleteResponse(task_id=completed.id, status=completed.status)
