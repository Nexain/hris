"""Knowledge-base lifecycle rules and a one-off reconcile job.

`supersede_replaced_versions` is the single rule for which versions a live version
replaces; `/publish` and the reconcile job both use it.

`KnowledgeBaseReconciler` repairs data written before those rules existed:
- several PUBLISHED versions of one document (keeps the newest),
- older versions that never went live (e.g. stuck in PROCESSING),
- `current_version_id` that does not point at the live version,
- published chunks built by an older chunker (re-processes them; they stay live).
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field

from app.models.document import AuditLog, Document, DocumentVersion, VersionStatus
from app.repositories.firestore import FirestoreRepository
from app.services.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

# Versions that never went live. When a newer version is live they are replaced.
NOT_YET_LIVE_STATUSES = {
    VersionStatus.DRAFT,
    VersionStatus.PROCESSING,
    VersionStatus.FAILED,
    VersionStatus.REVIEW,
}


def _version_sort_key(v: DocumentVersion) -> datetime:
    return v.published_at or v.created_at


def versions_replaced_by(live: DocumentVersion, versions: List[DocumentVersion]) -> List[DocumentVersion]:
    """Versions that `live` supersedes: any other PUBLISHED version, and older versions that
    never went live. Newer pending versions are kept so an upload in progress is not lost."""
    replaced = []
    for v in versions:
        if v.id == live.id:
            continue
        if v.status == VersionStatus.PUBLISHED:
            replaced.append(v)
        elif v.status in NOT_YET_LIVE_STATUSES and v.created_at < live.created_at:
            replaced.append(v)
    return replaced


async def supersede_replaced_versions(
    repo: FirestoreRepository,
    live: DocumentVersion,
    actor_id: str,
    versions: Optional[List[DocumentVersion]] = None,
    dry_run: bool = False,
) -> List[dict]:
    """Mark versions replaced by `live` as SUPERSEDED, with an audit entry each.

    Returns [{"version_id", "version", "previous_status"}] for every version changed
    (or that would change, when dry_run is set).
    """
    if versions is None:
        versions = await repo.get_document_versions(live.document_id)

    changed = []
    for v in versions_replaced_by(live, versions):
        changed.append({"version_id": v.id, "version": v.version, "previous_status": v.status.value})
        if dry_run:
            continue
        previous_state = v.status.value
        v.status = VersionStatus.SUPERSEDED
        await repo.update_document_version(v)
        await repo.log_audit(
            AuditLog(
                id=str(uuid.uuid4()),
                actor_id=actor_id,
                action="SUPERSEDE_VERSION",
                resource_type="document_version",
                resource_id=v.id,
                previous_state=previous_state,
                new_state=VersionStatus.SUPERSEDED.value,
                metadata={"superseded_by": live.id},
            )
        )
    return changed


class ReprocessResult(BaseModel):
    version_id: str
    chunk_count: Optional[int] = None
    sections: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class DocumentReconcileReport(BaseModel):
    document_id: str
    title: str
    live_version_id: Optional[str] = None
    current_version_id_before: Optional[str] = None
    current_version_id_after: Optional[str] = None
    superseded: List[dict] = Field(default_factory=list)
    reprocessed: List[ReprocessResult] = Field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(
            self.superseded
            or self.reprocessed
            or self.current_version_id_before != self.current_version_id_after
        )


class ReconcileReport(BaseModel):
    dry_run: bool
    reprocess: bool
    documents_checked: int = 0
    documents_changed: int = 0
    versions_superseded: int = 0
    current_version_ids_fixed: int = 0
    versions_reprocessed: int = 0
    reprocess_errors: int = 0
    documents: List[DocumentReconcileReport] = Field(default_factory=list)


class KnowledgeBaseReconciler:
    """Brings stored documents and versions in line with the lifecycle rules."""

    def __init__(self, repo: FirestoreRepository, processor: DocumentProcessor):
        self.repo = repo
        self.processor = processor

    async def reconcile(self, actor_id: str, dry_run: bool = True, reprocess: bool = True) -> ReconcileReport:
        report = ReconcileReport(dry_run=dry_run, reprocess=reprocess)
        for doc in await self.repo.list_documents():
            doc_report = await self._reconcile_document(doc, actor_id, dry_run, reprocess)
            report.documents_checked += 1
            report.versions_superseded += len(doc_report.superseded)
            if doc_report.current_version_id_before != doc_report.current_version_id_after:
                report.current_version_ids_fixed += 1
            report.versions_reprocessed += sum(1 for r in doc_report.reprocessed if not r.error)
            report.reprocess_errors += sum(1 for r in doc_report.reprocessed if r.error)
            if doc_report.changed:
                report.documents_changed += 1
                report.documents.append(doc_report)

        if not dry_run:
            await self.repo.log_audit(
                AuditLog(
                    id=str(uuid.uuid4()),
                    actor_id=actor_id,
                    action="RECONCILE_KNOWLEDGE_BASE",
                    resource_type="knowledge_base",
                    resource_id="all",
                    metadata=report.model_dump(exclude={"documents"}),
                )
            )
        return report

    async def _reconcile_document(
        self, doc: Document, actor_id: str, dry_run: bool, reprocess: bool
    ) -> DocumentReconcileReport:
        versions = list(doc.versions)
        published = sorted(
            (v for v in versions if v.status == VersionStatus.PUBLISHED), key=_version_sort_key
        )
        live = published[-1] if published else None

        result = DocumentReconcileReport(
            document_id=doc.id,
            title=doc.title,
            live_version_id=live.id if live else None,
            current_version_id_before=doc.current_version_id,
            current_version_id_after=live.id if live else None,
        )

        if live:
            result.superseded = await supersede_replaced_versions(
                self.repo, live, actor_id, versions=versions, dry_run=dry_run
            )

        if result.current_version_id_after != doc.current_version_id and not dry_run:
            doc.current_version_id = result.current_version_id_after
            doc.updated_at = datetime.now(timezone.utc)
            # Versions live in their own collection; get_document re-attaches them.
            await self.repo.create_document(doc.model_copy(update={"versions": []}))
            await self.repo.log_audit(
                AuditLog(
                    id=str(uuid.uuid4()),
                    actor_id=actor_id,
                    action="SET_CURRENT_VERSION",
                    resource_type="document",
                    resource_id=doc.id,
                    previous_state=result.current_version_id_before,
                    new_state=result.current_version_id_after,
                )
            )

        if live and reprocess:
            result.reprocessed.append(await self._reprocess(live.id, dry_run))
        return result

    async def _reprocess(self, version_id: str, dry_run: bool) -> ReprocessResult:
        if dry_run:
            return ReprocessResult(version_id=version_id)
        try:
            processed = await self.processor.process_version(version_id)
        except Exception as e:  # keep going; one bad PDF must not stop the rest
            logger.error(f"Reconcile: re-processing '{version_id}' failed: {e}")
            return ReprocessResult(version_id=version_id, error=str(e))
        sections: List[str] = []
        for chunk in processed.chunks:
            if chunk.section and chunk.section not in sections:
                sections.append(chunk.section)
        return ReprocessResult(version_id=version_id, chunk_count=processed.chunk_count, sections=sections)
