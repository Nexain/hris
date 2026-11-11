from typing import List

from fastapi import APIRouter, Depends, Query

from app.core.config import Settings, get_settings
from app.core.security import UserContext, hr_admin_ids, require_hr_admin
from app.models.onboarding import EmployeeSummary, OnboardingProgress, OnboardingTask
from app.repositories.firestore import FirestoreRepository, get_firestore_repository
from app.services.document_processor import DocumentProcessor, get_document_processor
from app.services.maintenance import KnowledgeBaseReconciler, ReconcileReport
from app.services.onboarding import OnboardingService, get_onboarding_service

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post(
    "/knowledge-base/reconcile",
    response_model=ReconcileReport,
    summary="Repair document lifecycle data (HR admin)",
    description=(
        "For every document: keep the newest PUBLISHED version live and supersede the rest, "
        "supersede older versions that never went live (e.g. stuck in PROCESSING), point "
        "current_version_id at the live version, and optionally re-process live versions with "
        "the current chunker. Defaults to a dry run that only reports what would change."
    ),
)
async def reconcile_knowledge_base(
    dry_run: bool = Query(True, description="Only report changes; set to false to apply them"),
    reprocess: bool = Query(True, description="Re-process each live version (calls the embedding model)"),
    current_user: UserContext = Depends(require_hr_admin),
    repo: FirestoreRepository = Depends(get_firestore_repository),
    processor: DocumentProcessor = Depends(get_document_processor),
):
    reconciler = KnowledgeBaseReconciler(repo=repo, processor=processor)
    return await reconciler.reconcile(actor_id=current_user.user_id, dry_run=dry_run, reprocess=reprocess)


# ---------------------------------------------------------------- HR onboarding monitor
# Read-only views of other employees' onboarding. HR cannot complete tasks for them:
# POST /tasks/{id}/complete only works for the task's owner.


@router.get(
    "/employees",
    response_model=List[EmployeeSummary],
    summary="List employees with onboarding progress (HR admin)",
)
async def list_employees(
    current_user: UserContext = Depends(require_hr_admin),
    service: OnboardingService = Depends(get_onboarding_service),
    settings: Settings = Depends(get_settings),
):
    return await service.list_employee_summaries(exclude_user_ids=hr_admin_ids(settings))


@router.get(
    "/employees/{user_id}/tasks",
    response_model=List[OnboardingTask],
    summary="An employee's onboarding tasks, read-only (HR admin)",
)
async def get_employee_tasks(
    user_id: str,
    current_user: UserContext = Depends(require_hr_admin),
    service: OnboardingService = Depends(get_onboarding_service),
):
    return await service.get_tasks(user_id)


@router.get(
    "/employees/{user_id}/progress",
    response_model=OnboardingProgress,
    summary="An employee's onboarding progress (HR admin)",
)
async def get_employee_progress(
    user_id: str,
    current_user: UserContext = Depends(require_hr_admin),
    service: OnboardingService = Depends(get_onboarding_service),
):
    return await service.get_progress(user_id)
