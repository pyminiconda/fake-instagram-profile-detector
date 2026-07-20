"""
schemas/users.py — Pydantic models for user management endpoints.
"""

from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, List


# ── Self-management (own profile) ─────────────────────────────────────

class UpdateUsernameRequest(BaseModel):
    username: str

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters.")
        if len(v) > 30:
            raise ValueError("Username must be 30 characters or fewer.")
        if not v.replace("_", "").isalnum():
            raise ValueError("Username may only contain letters, numbers, and underscores.")
        return v


class UpdateEmailRequest(BaseModel):
    email: EmailStr
    current_password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str
    confirm_password: str

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("New password must be at least 6 characters.")
        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        if "new_password" in info.data and v != info.data["new_password"]:
            raise ValueError("Passwords do not match.")
        return v


class DeleteAccountRequest(BaseModel):
    password: str


# ── Admin user management ─────────────────────────────────────────────

class CreateUserRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    is_admin: bool = False

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters.")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters.")
        return v


class UpdateRoleRequest(BaseModel):
    is_admin: bool


class AdminResetPasswordRequest(BaseModel):
    new_password: str

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters.")
        return v


# ── Responses ─────────────────────────────────────────────────────────

class UserDetailResponse(BaseModel):
    userId: str
    username: str
    email: str
    is_admin: bool
    createdAt: str
    lastLogin: Optional[str] = None


class UserListResponse(BaseModel):
    users: List[UserDetailResponse]
    total: int


class UserStatsResponse(BaseModel):
    total_scans: int
    fake_found: int
    genuine_found: int
    member_since: str
