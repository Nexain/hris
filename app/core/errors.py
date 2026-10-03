from typing import Any, Dict, Optional
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse


class AppException(HTTPException):
    """Custom application exception with structured error payload."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.message = message
        self.details = details or {}


class DuplicateDocumentException(AppException):
    """Raised when duplicate file or content is detected."""

    def __init__(self, message: str = "An identical document already exists."):
        super().__init__(
            code="DUPLICATE_DOCUMENT",
            message=message,
            status_code=status.HTTP_409_CONFLICT,
        )


class DuplicateProfileException(AppException):
    """Raised when attempting to create a profile that already exists."""

    def __init__(self, user_id: str):
        super().__init__(
            code="DUPLICATE_PROFILE",
            message=f"Profile for user '{user_id}' already exists.",
            status_code=status.HTTP_409_CONFLICT,
        )


class ResourceNotFoundException(AppException):
    """Raised when a requested resource is not found."""

    def __init__(self, resource: str, identifier: str):
        super().__init__(
            code="RESOURCE_NOT_FOUND",
            message=f"{resource} with id '{identifier}' not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class UnauthorizedAccessException(AppException):
    """Raised when an operation is unauthorized."""

    def __init__(self, message: str = "Access unauthorized."):
        super().__init__(
            code="UNAUTHORIZED",
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
        )


class DocumentProcessingException(AppException):
    """Raised when document extraction, parsing, or chunking fails."""

    def __init__(self, message: str):
        super().__init__(
            code="DOCUMENT_PROCESSING_FAILED",
            message=message,
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
        )


class InvalidStateTransitionException(AppException):
    """Raised when a document version lifecycle transition is not allowed."""

    def __init__(self, message: str = "Invalid document version state transition."):
        super().__init__(
            code="INVALID_STATE_TRANSITION",
            message=message,
            status_code=status.HTTP_409_CONFLICT,
        )


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """FastAPI exception handler for AppException instances."""
    content: Dict[str, Any] = {
        "error": {
            "code": exc.code,
            "message": exc.message,
        }
    }
    if exc.details:
        content["error"]["details"] = exc.details
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
    )
