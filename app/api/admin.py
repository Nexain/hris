from fastapi import APIRouter, Depends, Query

from app.core.security import UserContext, require_hr_admin
from app.repositories.firestore import FirestoreRepository, get_firestore_repository
from app.services.document_processor import DocumentProcessor, get_document_processor
from app.services.maintenance import KnowledgeBaseReconciler, ReconcileReport

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
