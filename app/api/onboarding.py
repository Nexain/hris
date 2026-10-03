from fastapi import APIRouter, Depends, status

from app.core.security import UserContext, get_current_user
from app.models.onboarding import (
    OnboardingProgress,
    UserProfile,
    UserProfileCreate,
    UserProfileUpdate,
)
from app.services.onboarding import OnboardingService, get_onboarding_service

router = APIRouter(tags=["Onboarding"])


@router.post(
    "/profile",
    response_model=UserProfile,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new employee profile",
    description="Registers a new employee profile and generates their personalized onboarding tasks.",
)
async def create_profile(
    profile_in: UserProfileCreate,
    current_user: UserContext = Depends(get_current_user),
    service: OnboardingService = Depends(get_onboarding_service),
):
    """Create a new employee profile with onboarding tasks."""
    return await service.create_profile(
        user_id=profile_in.user_id or current_user.user_id,
        name=profile_in.name,
        role=profile_in.role,
        department=profile_in.department,
        location=profile_in.location,
    )


@router.get(
    "/profile",
    response_model=UserProfile,
    summary="Get current user profile",
    description="Retrieve the employee's onboarding profile. Returns 404 if not registered.",
)
async def get_profile(
    current_user: UserContext = Depends(get_current_user),
    service: OnboardingService = Depends(get_onboarding_service),
):
    """Retrieve profile for the authenticated employee."""
    return await service.get_profile(user_id=current_user.user_id)


@router.put(
    "/profile",
    response_model=UserProfile,
    summary="Update user profile",
    description="Update employee profile details such as role, department, or office location.",
)
async def update_profile(
    updates: UserProfileUpdate,
    current_user: UserContext = Depends(get_current_user),
    service: OnboardingService = Depends(get_onboarding_service),
):
    """Update employee profile."""
    return await service.update_profile(current_user.user_id, updates)


@router.get(
    "/onboarding/progress",
    response_model=OnboardingProgress,
    summary="Get onboarding progress",
    description="Calculates onboarding completion rate, total tasks, and remaining tasks.",
)
async def get_progress(
    current_user: UserContext = Depends(get_current_user),
    service: OnboardingService = Depends(get_onboarding_service),
):
    """Get onboarding completion progress for the current employee."""
    return await service.get_progress(current_user.user_id)
