"""Simple role-based access policy for retrieval-time hydration."""

from __future__ import annotations


AUTHORIZED_ROLES = {"doctor"}
REDACTED_ROLES = {"receptionist"}
ADMIN_ROLES = {"admin"}


def role_is_authorized(role: str | None) -> bool:
    normalized = (role or "").strip().lower()
    return normalized in AUTHORIZED_ROLES


def role_is_admin(role: str | None) -> bool:
    normalized = (role or "").strip().lower()
    return normalized in ADMIN_ROLES
