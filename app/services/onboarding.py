import uuid
from datetime import datetime, timezone
from typing import List, Optional

from app.core.errors import ResourceNotFoundException, UnauthorizedAccessException
from app.models.onboarding import (
    OnboardingProgress,
    OnboardingTask,
    TaskStatus,
    UserProfile,
    UserProfileUpdate,
)
from app.repositories.firestore import FirestoreRepository, get_firestore_repository


class OnboardingService:
    """Service handling employee profile, personalized task generation, and progress tracking."""

    def __init__(self, firestore_repo: Optional[FirestoreRepository] = None):
        self.firestore_repo = firestore_repo or get_firestore_repository()

    async def get_or_create_profile(
        self,
        user_id: str,
        name: str = "Aji",
        role: str = "Backend Engineer",
        department: str = "Engineering",
        location: str = "Jakarta",
    ) -> UserProfile:
        """Fetch user profile or initialize profile with personalized onboarding tasks."""
        profile = await self.firestore_repo.get_user(user_id)
        if not profile:
            profile = UserProfile(
                user_id=user_id,
                name=name,
                role=role,
                department=department,
                location=location,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            await self.firestore_repo.save_user(profile)
            # Generate default personalized tasks
            await self.generate_personalized_tasks(user_id, role, department, location)

        return profile

    async def update_profile(self, user_id: str, updates: UserProfileUpdate) -> UserProfile:
        """Update employee profile fields."""
        profile = await self.get_or_create_profile(user_id)

        if updates.name is not None:
            profile.name = updates.name
        if updates.role is not None:
            profile.role = updates.role
        if updates.department is not None:
            profile.department = updates.department
        if updates.location is not None:
            profile.location = updates.location

        profile.updated_at = datetime.now(timezone.utc)
        return await self.firestore_repo.save_user(profile)

    async def generate_personalized_tasks(
        self,
        user_id: str,
        role: str,
        department: str,
        location: str,
    ) -> List[OnboardingTask]:
        """Generate tailored onboarding checklist based on role, department, and location."""
        existing_tasks = await self.firestore_repo.get_tasks(user_id)
        if existing_tasks:
            return existing_tasks

        tasks: List[OnboardingTask] = []

        # Day 1: General & Hardware
        tasks.append(
            OnboardingTask(
                id=f"{user_id}_task_001",
                user_id=user_id,
                title="Read Employee Handbook",
                description=f"Review company values, code of conduct, and {location} office guidelines.",
                day=1,
                source_document_id="employee-handbook",
            )
        )
        tasks.append(
            OnboardingTask(
                id=f"{user_id}_task_002",
                user_id=user_id,
                title="Request laptop and peripherals",
                description="Submit equipment request according to the IT hardware policy.",
                day=1,
                source_document_id="laptop-request-procedure",
            )
        )
        tasks.append(
            OnboardingTask(
                id=f"{user_id}_task_003",
                user_id=user_id,
                title="Set up company Google Workspace and Slack accounts",
                description="Configure 2FA security and log into corporate email and Slack channels.",
                day=1,
                source_document_id="it-security-guide",
            )
        )

        # Day 2: Security & Department Specific
        tasks.append(
            OnboardingTask(
                id=f"{user_id}_task_004",
                user_id=user_id,
                title="Review IT Security & Data Privacy Guide",
                description="Understand data protection, password management, and compliance rules.",
                day=2,
                source_document_id="it-security-guide",
            )
        )

        if "engineer" in role.lower() or "tech" in department.lower() or "engineering" in department.lower():
            tasks.append(
                OnboardingTask(
                    id=f"{user_id}_task_005",
                    user_id=user_id,
                    title="Configure development environment and Git repositories",
                    description="Clone engineering repositories, set up SSH keys, and run local test suites.",
                    day=2,
                    source_document_id="engineering-onboarding-guide",
                )
            )
        else:
            tasks.append(
                OnboardingTask(
                    id=f"{user_id}_task_005",
                    user_id=user_id,
                    title="Review departmental onboarding materials",
                    description=f"Read operational guides and documentation for {department}.",
                    day=2,
                )
            )

        # Day 3: Team Alignment
        tasks.append(
            OnboardingTask(
                id=f"{user_id}_task_006",
                user_id=user_id,
                title="Meet assigned peer buddy and direct manager",
                description="Schedule a 1-on-1 welcome session to discuss team goals and first week milestones.",
                day=3,
            )
        )
        tasks.append(
            OnboardingTask(
                id=f"{user_id}_task_007",
                user_id=user_id,
                title=f"Review {location} office leave and attendance procedures",
                description=f"Understand time-off logging and remote work policies for {location}.",
                day=3,
                source_document_id="leave-policy",
            )
        )

        await self.firestore_repo.save_tasks(tasks)
        return tasks

    async def get_tasks(self, user_id: str) -> List[OnboardingTask]:
        """Fetch all onboarding tasks for user (auto-creating if needed)."""
        await self.get_or_create_profile(user_id)
        return await self.firestore_repo.get_tasks(user_id)

    async def complete_task(self, user_id: str, task_id: str) -> OnboardingTask:
        """Mark task as completed, verifying user ownership."""
        task = await self.firestore_repo.get_task(task_id)
        if not task:
            raise ResourceNotFoundException("OnboardingTask", task_id)

        # Security boundary: user can only complete their own tasks (Section 13.1)
        if task.user_id != user_id:
            raise UnauthorizedAccessException("You cannot modify another user's onboarding tasks.")

        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now(timezone.utc)
        return await self.firestore_repo.update_task(task)

    async def get_progress(self, user_id: str) -> OnboardingProgress:
        """Calculate onboarding completion progress from persisted task states."""
        tasks = await self.get_tasks(user_id)
        total = len(tasks)
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        remaining = total - completed
        percent = round((completed / total) * 100, 2) if total > 0 else 0.0

        return OnboardingProgress(
            total=total,
            completed=completed,
            remaining=remaining,
            progress_percent=percent,
        )


_onboarding_service_instance: Optional[OnboardingService] = None


def get_onboarding_service() -> OnboardingService:
    """Dependency provider for OnboardingService."""
    global _onboarding_service_instance
    if _onboarding_service_instance is None:
        _onboarding_service_instance = OnboardingService()
    return _onboarding_service_instance
