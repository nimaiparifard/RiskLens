"""Role-based permissions: viewer (read) < officer (score / decide) < admin (override, models)."""

from rest_framework.permissions import SAFE_METHODS, BasePermission


def role_of(user) -> str | None:
    return getattr(user, "role", None) if user and user.is_authenticated else None


class IsOfficer(BasePermission):
    message = "Credit officer or admin role required."

    def has_permission(self, request, view) -> bool:
        return role_of(request.user) in ("officer", "admin")


class IsAdminRole(BasePermission):
    message = "Admin role required."

    def has_permission(self, request, view) -> bool:
        return role_of(request.user) == "admin"


class ReadOnlyOrOfficer(BasePermission):
    """Everyone authenticated can read; writing needs officer/admin."""

    message = "Credit officer or admin role required."

    def has_permission(self, request, view) -> bool:
        if not (request.user and request.user.is_authenticated):
            return False
        return request.method in SAFE_METHODS or role_of(request.user) in ("officer", "admin")
