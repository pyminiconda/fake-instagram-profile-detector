"""
routers/users.py — User self-management endpoints.

GET    /api/users/me              — Own profile + stats
PUT    /api/users/me/username     — Change username
PUT    /api/users/me/email        — Change email
PUT    /api/users/me/password     — Change password
DELETE /api/users/me              — Delete own account
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import bcrypt
from fastapi import APIRouter, HTTPException, Depends, status

from backend.middleware.auth import get_current_user
from backend.schemas.users import (
    UpdateUsernameRequest, UpdateEmailRequest, ChangePasswordRequest,
    DeleteAccountRequest, UserDetailResponse, UserStatsResponse,
)
from backend.core.database import DatabaseManager

router = APIRouter(prefix="/api/users", tags=["User Profile"])


def get_db() -> DatabaseManager:
    from backend.main import app
    return app.state.db


def _verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── Own profile ───────────────────────────────────────────────────────

@router.get("/me", response_model=UserDetailResponse)
def get_me(
    current_user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    user = db.get_user_by_id(current_user["sub"])
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return UserDetailResponse(
        userId=user["userId"],
        username=user["username"],
        email=user["email"],
        is_admin=bool(user["is_admin"]),
        createdAt=user["createdAt"],
        lastLogin=user.get("lastLogin"),
    )


@router.get("/me/stats", response_model=UserStatsResponse)
def get_my_stats(
    current_user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    user = db.get_user_by_id(current_user["sub"])
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    records = db.get_history(current_user["sub"])
    fake_found = sum(1 for r in records if r.get("resultLabel") == "fake")
    genuine_found = sum(1 for r in records if r.get("resultLabel") == "genuine")

    return UserStatsResponse(
        total_scans=len(records),
        fake_found=fake_found,
        genuine_found=genuine_found,
        member_since=user["createdAt"],
    )


# ── Update username ───────────────────────────────────────────────────

@router.put("/me/username", response_model=UserDetailResponse)
def update_username(
    body: UpdateUsernameRequest,
    current_user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    dupes = db.check_duplicate(username=body.username)
    if dupes["username_exists"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Username already taken.")

    with db._get_connection() as conn:
        conn.execute(
            "UPDATE USERS SET username = ? WHERE userId = ?",
            (body.username, current_user["sub"]),
        )

    user = db.get_user_by_id(current_user["sub"])
    return UserDetailResponse(
        userId=user["userId"],
        username=user["username"],
        email=user["email"],
        is_admin=bool(user["is_admin"]),
        createdAt=user["createdAt"],
        lastLogin=user.get("lastLogin"),
    )


# ── Update email ──────────────────────────────────────────────────────

@router.put("/me/email", response_model=UserDetailResponse)
def update_email(
    body: UpdateEmailRequest,
    current_user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    user = db.get_user_by_id(current_user["sub"])
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    if not _verify_password(body.current_password, user["hashedPassword"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Current password is incorrect.")

    dupes = db.check_duplicate(email=str(body.email))
    if dupes["email_exists"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Email already registered.")

    with db._get_connection() as conn:
        conn.execute(
            "UPDATE USERS SET email = ? WHERE userId = ?",
            (str(body.email), current_user["sub"]),
        )

    updated = db.get_user_by_id(current_user["sub"])
    return UserDetailResponse(
        userId=updated["userId"],
        username=updated["username"],
        email=updated["email"],
        is_admin=bool(updated["is_admin"]),
        createdAt=updated["createdAt"],
        lastLogin=updated.get("lastLogin"),
    )


# ── Change password ───────────────────────────────────────────────────

@router.put("/me/password")
def change_password(
    body: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    user = db.get_user_by_id(current_user["sub"])
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    if not _verify_password(body.current_password, user["hashedPassword"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Current password is incorrect.")

    db.update_password(current_user["sub"], body.new_password)
    return {"message": "Password updated successfully."}


# ── Delete account ────────────────────────────────────────────────────

@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
def delete_account(
    body: DeleteAccountRequest,
    current_user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    user = db.get_user_by_id(current_user["sub"])
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    if not _verify_password(body.password, user["hashedPassword"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect password.")

    with db._get_connection() as conn:
        conn.execute("DELETE FROM SEARCH_HISTORY WHERE userId = ?", (current_user["sub"],))
        conn.execute("DELETE FROM USERS WHERE userId = ?", (current_user["sub"],))
