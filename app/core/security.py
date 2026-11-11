from enum import Enum
from typing import Optional, Set

from fastapi import Depends, Header
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.errors import UnauthorizedAccessException


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


def hr_admin_ids(settings: Settings) -> Set[str]:
    """Parse the configured HR admin allowlist."""
    return {uid.strip() for uid in settings.hr_admin_user_ids.split(",") if uid.strip()}


def get_current_user(
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_user_department: Optional[str] = Header(default=None, alias="X-User-Department"),
    x_user_location: Optional[str] = Header(default=None, alias="X-User-Location"),
    settings: Settings = Depends(get_settings),
) -> UserContext:
    """Build the caller's context from request headers.

    The access level is decided on the server from the HR admin allowlist. A client
    cannot raise its own privileges by sending ``X-Access-Level``; that header is
    ignored.
    """
    user_id = (x_user_id or "user_001").strip()

    if user_id in hr_admin_ids(settings):
        return UserContext(
            user_id=user_id,
            name="HR Admin",
            role=x_user_role or "People Operations",
            department=x_user_department or "People Operations",
            location=x_user_location or "Jakarta",
            access_level=AccessLevel.HR_ADMIN,
        )

    return UserContext(
        user_id=user_id,
        name="Aji" if user_id == "user_001" else user_id,
        role=x_user_role or "Backend Engineer",
        department=x_user_department or "Engineering",
        location=x_user_location or "Jakarta",
        access_level=AccessLevel.EMPLOYEE,
    )


def require_hr_admin(current_user: UserContext = Depends(get_current_user)) -> UserContext:
    """Allow only HR admins; everyone else gets 403."""
    if not current_user.is_hr_admin():
        raise UnauthorizedAccessException("This action requires HR admin access.")
    return current_user
