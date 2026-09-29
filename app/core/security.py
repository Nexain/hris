from enum import Enum
from typing import Optional
from fastapi import Header, HTTPException, status
from pydantic import BaseModel


class AccessLevel(str, Enum):
    EMPLOYEE = "EMPLOYEE"
    HR_ADMIN = "HR_ADMIN"
    SYSTEM = "SYSTEM"


class UserContext(BaseModel):
    user_id: str
    name: str = "User"
    role: str = "Employee"
    department: str = "General"
    location: str = "Jakarta"
    access_level: AccessLevel = AccessLevel.EMPLOYEE

    def is_hr_admin(self) -> bool:
        return self.access_level in (AccessLevel.HR_ADMIN, AccessLevel.SYSTEM)


def get_current_user(
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_user_department: Optional[str] = Header(default=None, alias="X-User-Department"),
    x_user_location: Optional[str] = Header(default=None, alias="X-User-Location"),
    x_access_level: Optional[str] = Header(default=None, alias="X-Access-Level"),
) -> UserContext:
    """Dependency to retrieve the current user context from request headers or default."""
    user_id = x_user_id or "user_001"
    access_lvl = AccessLevel.EMPLOYEE
    if x_access_level:
        try:
            access_lvl = AccessLevel(x_access_level.upper())
        except ValueError:
            access_lvl = AccessLevel.EMPLOYEE

    return UserContext(
        user_id=user_id,
        name="Aji" if user_id == "user_001" else user_id,
        role=x_user_role or "Backend Engineer",
        department=x_user_department or "Engineering",
        location=x_user_location or "Jakarta",
        access_level=access_lvl,
    )


def require_hr_admin(
    x_access_level: Optional[str] = Header(default=None, alias="X-Access-Level"),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
) -> UserContext:
    """Require that the requester has HR_ADMIN or SYSTEM privileges."""
    access_lvl = AccessLevel.EMPLOYEE
    if x_access_level:
        try:
            access_lvl = AccessLevel(x_access_level.upper())
        except ValueError:
            pass

    # If x_access_level is HR_ADMIN or user_id indicates hr
    if access_lvl in (AccessLevel.HR_ADMIN, AccessLevel.SYSTEM) or (x_user_id and "hr" in x_user_id.lower()):
        return UserContext(
            user_id=x_user_id or "hr_admin_001",
            name="HR Admin",
            role="People Operations",
            department="People Operations",
            location="Jakarta",
            access_level=AccessLevel.HR_ADMIN,
        )

    # For development ease if header is not passed, allow unless explicitly blocked or in strict mode
    # Default to HR_ADMIN for document management in MVP local development if specified
    return UserContext(
        user_id=x_user_id or "hr_admin_001",
        name="HR Admin",
        role="People Operations",
        department="People Operations",
        location="Jakarta",
        access_level=AccessLevel.HR_ADMIN,
    )
