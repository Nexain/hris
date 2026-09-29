import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.config import Settings, get_settings
from app.models.document import (
    AuditLog,
    Document,
    DocumentChunk,
    DocumentType,
    DocumentVersion,
    EvaluationQuestion,
    VersionStatus,
)
from app.models.onboarding import OnboardingTask, UserProfile

logger = logging.getLogger(__name__)


class FirestoreRepository:
    """Repository handling persistence in Google Cloud Firestore with in-memory fallback."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._client = None
        self._use_in_memory = False

        # In-memory storage for offline / testing / fallback
        self._mem_documents: Dict[str, Dict[str, Any]] = {}
        self._mem_versions: Dict[str, Dict[str, Any]] = {}
        self._mem_chunks: Dict[str, Dict[str, Any]] = {}
        self._mem_audit_logs: List[Dict[str, Any]] = []
        self._mem_users: Dict[str, Dict[str, Any]] = {}
        self._mem_tasks: Dict[str, Dict[str, Any]] = {}
        self._mem_evaluation_questions: Dict[str, Dict[str, Any]] = {}

        if self.settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                self.settings.google_application_credentials
            )

    @property
    def client(self):
        """Lazy-initialize Google Cloud Firestore client or fallback to in-memory."""
        if self._client is None and not self._use_in_memory:
            project_id = self.settings.gcp_project_id.strip()
            if not project_id or project_id == "your-gcp-project-id":
                logger.info("No GCP Project ID configured; Firestore using in-memory store.")
                self._use_in_memory = True
                return None

            try:
                from google.cloud import firestore
                database = self.settings.firestore_database or "(default)"
                self._client = firestore.Client(project=project_id, database=database)
                logger.info(f"Connected to Firestore in project '{project_id}', database '{database}'.")
            except Exception as e:
                logger.warning(f"Could not initialize Firestore client: {e}. Falling back to in-memory store.")
                self._use_in_memory = True
                self._client = None
        return self._client

    # ------------------ Documents ------------------
    async def create_document(self, doc: Document) -> Document:
        """Create a new logical document record."""
        data = doc.model_dump(mode="json")
        if self.client:
            self.client.collection("documents").document(doc.id).set(data)
        else:
            self._mem_documents[doc.id] = data
        return doc

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Fetch a logical document by ID, along with its versions."""
        if self.client:
            doc_ref = self.client.collection("documents").document(document_id).get()
            if not doc_ref.exists:
                return None
            data = doc_ref.to_dict()
        else:
            data = self._mem_documents.get(document_id)
            if not data:
                return None

        # Fetch attached versions
        versions = await self.get_document_versions(document_id)
        data["versions"] = [v.model_dump(mode="json") for v in versions]
        return Document(**data)

    async def list_documents(
        self,
        department: Optional[str] = None,
        location: Optional[str] = None,
        document_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Document]:
        """List documents matching filter criteria."""
        results: List[Document] = []

        if self.client:
            query = self.client.collection("documents")
            if department:
                query = query.where("department", "==", department)
            if location:
                query = query.where("location", "==", location)
            if document_type:
                query = query.where("document_type", "==", document_type)

            for doc_snap in query.stream():
                data = doc_snap.to_dict()
                versions = await self.get_document_versions(data["id"])
                data["versions"] = [v.model_dump(mode="json") for v in versions]
                results.append(Document(**data))
        else:
            for d in self._mem_documents.values():
                if department and d.get("department") != department:
                    continue
                if location and d.get("location") != location:
                    continue
                if document_type and d.get("document_type") != document_type:
                    continue
                versions = await self.get_document_versions(d["id"])
                d_copy = dict(d)
                d_copy["versions"] = [v.model_dump(mode="json") for v in versions]
                results.append(Document(**d_copy))

        # Filter by version status if requested
        if status:
            filtered = []
            for doc in results:
                if any(v.status == status for v in doc.versions):
                    filtered.append(doc)
            return filtered

        return results

    # ------------------ Document Versions ------------------
    async def create_document_version(self, version: DocumentVersion) -> DocumentVersion:
        """Create a new document version."""
        data = version.model_dump(mode="json")
        if self.client:
            self.client.collection("document_versions").document(version.id).set(data)
        else:
            self._mem_versions[version.id] = data
        return version

    async def get_document_version(self, version_id: str) -> Optional[DocumentVersion]:
        """Fetch a specific version by version ID."""
        if self.client:
            doc_ref = self.client.collection("document_versions").document(version_id).get()
            if not doc_ref.exists:
                return None
            return DocumentVersion(**doc_ref.to_dict())
        else:
            data = self._mem_versions.get(version_id)
            return DocumentVersion(**data) if data else None

    async def get_document_versions(self, document_id: str) -> List[DocumentVersion]:
        """Fetch all versions of a document."""
        versions: List[DocumentVersion] = []
        if self.client:
            query = self.client.collection("document_versions").where("document_id", "==", document_id)
            for snap in query.stream():
                versions.append(DocumentVersion(**snap.to_dict()))
        else:
            for v in self._mem_versions.values():
                if v.get("document_id") == document_id:
                    versions.append(DocumentVersion(**v))
        versions.sort(key=lambda x: x.created_at)
        return versions

    async def update_document_version(self, version: DocumentVersion) -> DocumentVersion:
        """Update an existing document version."""
        data = version.model_dump(mode="json")
        if self.client:
            self.client.collection("document_versions").document(version.id).set(data, merge=True)
        else:
            self._mem_versions[version.id] = data
        return version

    async def find_version_by_file_hash(self, file_hash: str) -> Optional[DocumentVersion]:
        """Check if any version matches the binary SHA-256 hash."""
        if self.client:
            query = self.client.collection("document_versions").where("file_hash", "==", file_hash).limit(1)
            docs = list(query.stream())
            if docs:
                return DocumentVersion(**docs[0].to_dict())
            return None
        else:
            for v in self._mem_versions.values():
                if v.get("file_hash") == file_hash:
                    return DocumentVersion(**v)
            return None

    async def find_version_by_content_hash(self, content_hash: str) -> Optional[DocumentVersion]:
        """Check if any version matches the normalized content SHA-256 hash."""
        if self.client:
            query = self.client.collection("document_versions").where("content_hash", "==", content_hash).limit(1)
            docs = list(query.stream())
            if docs:
                return DocumentVersion(**docs[0].to_dict())
            return None
        else:
            for v in self._mem_versions.values():
                if v.get("content_hash") == content_hash:
                    return DocumentVersion(**v)
            return None

    # ------------------ Document Chunks ------------------
    async def save_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Persist a batch of document chunks."""
        if self.client:
            batch = self.client.batch()
            for chunk in chunks:
                ref = self.client.collection("document_chunks").document(chunk.id)
                batch.set(ref, chunk.model_dump(mode="json"))
            batch.commit()
        else:
            for chunk in chunks:
                self._mem_chunks[chunk.id] = chunk.model_dump(mode="json")

    async def get_chunks_by_version(self, version_id: str) -> List[DocumentChunk]:
        """Fetch all chunks for a given version."""
        chunks: List[DocumentChunk] = []
        if self.client:
            query = self.client.collection("document_chunks").where("version_id", "==", version_id)
            for snap in query.stream():
                chunks.append(DocumentChunk(**snap.to_dict()))
        else:
            for c in self._mem_chunks.values():
                if c.get("version_id") == version_id:
                    chunks.append(DocumentChunk(**c))
        chunks.sort(key=lambda x: x.chunk_index)
        return chunks

    async def get_published_chunks(
        self,
        department: Optional[str] = None,
        location: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Fetch all chunks belonging to PUBLISHED document versions, with optional metadata filters."""
        published_version_ids = set()
        if self.client:
            query = self.client.collection("document_versions").where("status", "==", VersionStatus.PUBLISHED.value)
            for snap in query.stream():
                published_version_ids.add(snap.id)
        else:
            for v in self._mem_versions.values():
                if v.get("status") in (VersionStatus.PUBLISHED.value, VersionStatus.PUBLISHED):
                    published_version_ids.add(v.get("id"))

        if not published_version_ids:
            return []

        chunks: List[DocumentChunk] = []
        if self.client:
            for vid in published_version_ids:
                q = self.client.collection("document_chunks").where("version_id", "==", vid)
                for snap in q.stream():
                    chunks.append(DocumentChunk(**snap.to_dict()))
        else:
            for c in self._mem_chunks.values():
                if c.get("version_id") in published_version_ids:
                    chunks.append(DocumentChunk(**c))

        # Department/location are soft scoping hints, not authorization boundaries.
        # All PUBLISHED chunks remain retrievable (spec §13.1: employees may access
        # published company documentation); callers may use these hints for ranking.
        # Unpublished versions are already excluded via published_version_ids above.
        return chunks

    async def delete_chunks_by_version(self, version_id: str) -> None:
        """Delete chunks for a given version (e.g. for reprocessing)."""
        if self.client:
            query = self.client.collection("document_chunks").where("version_id", "==", version_id)
            batch = self.client.batch()
            for snap in query.stream():
                batch.delete(snap.reference)
            batch.commit()
        else:
            keys_to_del = [cid for cid, c in self._mem_chunks.items() if c.get("version_id") == version_id]
            for cid in keys_to_del:
                del self._mem_chunks[cid]

    # ------------------ Audit Logs ------------------
    async def log_audit(self, audit: AuditLog) -> None:
        """Write an immutable audit log entry."""
        data = audit.model_dump(mode="json")
        if self.client:
            self.client.collection("audit_logs").document(audit.id).set(data)
        else:
            self._mem_audit_logs.append(data)

    # ------------------ Evaluation Questions ------------------
    async def save_evaluation_questions(self, questions: List[EvaluationQuestion]) -> None:
        """Persist a batch of regression evaluation questions."""
        if self.client:
            batch = self.client.batch()
            for q in questions:
                ref = self.client.collection("evaluation_questions").document(q.id)
                batch.set(ref, q.model_dump(mode="json"))
            batch.commit()
        else:
            for q in questions:
                self._mem_evaluation_questions[q.id] = q.model_dump(mode="json")

    async def get_evaluation_questions(self, document_id: str) -> List[EvaluationQuestion]:
        """Fetch persisted evaluation questions for a document."""
        questions: List[EvaluationQuestion] = []
        if self.client:
            query = self.client.collection("evaluation_questions").where("document_id", "==", document_id)
            for snap in query.stream():
                questions.append(EvaluationQuestion(**snap.to_dict()))
        else:
            for q in self._mem_evaluation_questions.values():
                if q.get("document_id") == document_id:
                    questions.append(EvaluationQuestion(**q))
        questions.sort(key=lambda x: x.created_at)
        return questions

    # ------------------ Users ------------------
    async def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Fetch a user profile by ID."""
        if self.client:
            snap = self.client.collection("users").document(user_id).get()
            if not snap.exists:
                return None
            return UserProfile(**snap.to_dict())
        else:
            data = self._mem_users.get(user_id)
            return UserProfile(**data) if data else None

    async def save_user(self, user: UserProfile) -> UserProfile:
        """Create or update a user profile."""
        data = user.model_dump(mode="json")
        if self.client:
            self.client.collection("users").document(user.user_id).set(data, merge=True)
        else:
            self._mem_users[user.user_id] = data
        return user

    # ------------------ Onboarding Tasks ------------------
    async def get_tasks(self, user_id: str) -> List[OnboardingTask]:
        """Fetch all onboarding tasks for a user, sorted by day and creation."""
        tasks: List[OnboardingTask] = []
        if self.client:
            q = self.client.collection("onboarding_tasks").where("user_id", "==", user_id)
            for snap in q.stream():
                tasks.append(OnboardingTask(**snap.to_dict()))
        else:
            for t in self._mem_tasks.values():
                if t.get("user_id") == user_id:
                    tasks.append(OnboardingTask(**t))
        tasks.sort(key=lambda x: (x.day, x.id))
        return tasks

    async def get_task(self, task_id: str) -> Optional[OnboardingTask]:
        """Fetch an onboarding task by ID."""
        if self.client:
            snap = self.client.collection("onboarding_tasks").document(task_id).get()
            if not snap.exists:
                return None
            return OnboardingTask(**snap.to_dict())
        else:
            data = self._mem_tasks.get(task_id)
            return OnboardingTask(**data) if data else None

    async def save_tasks(self, tasks: List[OnboardingTask]) -> None:
        """Persist a list of onboarding tasks."""
        if self.client:
            batch = self.client.batch()
            for t in tasks:
                ref = self.client.collection("onboarding_tasks").document(t.id)
                batch.set(ref, t.model_dump(mode="json"))
            batch.commit()
        else:
            for t in tasks:
                self._mem_tasks[t.id] = t.model_dump(mode="json")

    async def update_task(self, task: OnboardingTask) -> OnboardingTask:
        """Update an existing task."""
        data = task.model_dump(mode="json")
        if self.client:
            self.client.collection("onboarding_tasks").document(task.id).set(data, merge=True)
        else:
            self._mem_tasks[task.id] = data
        return task


_firestore_repo_instance: Optional[FirestoreRepository] = None


def get_firestore_repository() -> FirestoreRepository:
    """Dependency provider for FirestoreRepository."""
    global _firestore_repo_instance
    if _firestore_repo_instance is None:
        _firestore_repo_instance = FirestoreRepository()
    return _firestore_repo_instance
