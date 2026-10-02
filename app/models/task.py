"""Task models (spec §8 lists models/task.py).

Task models are defined alongside onboarding models; this module re-exports them
so the spec's file layout is satisfied without duplicating definitions.
"""

from app.models.onboarding import (
    OnboardingProgress,
    OnboardingTask,
    TaskCompleteResponse,
    TaskStatus,
)

__all__ = [
    "OnboardingTask",
    "OnboardingProgress",
    "TaskCompleteResponse",
    "TaskStatus",
]
