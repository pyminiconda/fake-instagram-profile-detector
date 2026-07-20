"""
routers/auth.py — Authentication endpoints.

POST /api/auth/login
POST /api/auth/signup
POST /api/auth/refresh
GET  /api/auth/me
POST /api/auth/forgot-password
POST /api/auth/reset-password
POST /api/auth/logout
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException, Depends, status

from backend.schemas.auth import (
    LoginRequest, SignupRequest, AuthResponse, UserResponse,
    RefreshRequest, ForgotPasswordRequest, ResetPasswordRequest, MessageResponse,
    TokenResponse,
)
from backend.middleware.auth import (
    create_access_token, create_refresh_token, decode_refresh_token,
    get_current_user, create_reset_token, verify_reset_token,
)
from backend.core.database import DatabaseManager

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# Shared DB instance (injected from main.py via app.state)
def get_db() -> DatabaseManager:
    from backend.main import app
    return app.state.db


# ── Login ─────────────────────────────────────────────────────────────

@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest, db: DatabaseManager = Depends(get_db)):
    user = db.authenticate_user(body.email, body.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid email or password.")

    access_token  = create_access_token(user["userId"], user["username"], bool(user["is_admin"]))
    refresh_token = create_refresh_token(user["userId"])

    return AuthResponse(
        user=UserResponse(
            userId=user["userId"],
            username=user["username"],
            email=user["email"],
            is_admin=bool(user["is_admin"]),
            createdAt=user["createdAt"],
            lastLogin=user.get("lastLogin"),
        ),
        access_token=access_token,
        refresh_token=refresh_token,
    )


# ── Signup ────────────────────────────────────────────────────────────

@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def signup(body: SignupRequest, db: DatabaseManager = Depends(get_db)):
    dupes = db.check_duplicate(username=body.username, email=body.email)
    errors = []
    if dupes["username_exists"]:
        errors.append("Username already taken.")
    if dupes["email_exists"]:
        errors.append("Email already registered.")
    if errors:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=" ".join(errors))

    try:
        new_user = db.create_user(body.username, str(body.email), body.password)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    access_token  = create_access_token(new_user["userId"], new_user["username"], False)
    refresh_token = create_refresh_token(new_user["userId"])

    full_user = db.get_user_by_id(new_user["userId"])
    return AuthResponse(
        user=UserResponse(
            userId=full_user["userId"],
            username=full_user["username"],
            email=full_user["email"],
            is_admin=bool(full_user.get("is_admin", 0)),
            createdAt=full_user["createdAt"],
            lastLogin=full_user.get("lastLogin"),
        ),
        access_token=access_token,
        refresh_token=refresh_token,
    )


# ── Refresh token ─────────────────────────────────────────────────────

@router.post("/refresh", response_model=TokenResponse)
def refresh_token(body: RefreshRequest, db: DatabaseManager = Depends(get_db)):
    user_id = decode_refresh_token(body.refresh_token)
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")

    access_token  = create_access_token(user["userId"], user["username"], bool(user["is_admin"]))
    refresh_token_new = create_refresh_token(user["userId"])
    return TokenResponse(access_token=access_token, refresh_token=refresh_token_new)


# ── Me ────────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserResponse)
def me(current_user: dict = Depends(get_current_user),
       db: DatabaseManager = Depends(get_db)):
    user = db.get_user_by_id(current_user["sub"])
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return UserResponse(
        userId=user["userId"],
        username=user["username"],
        email=user["email"],
        is_admin=bool(user["is_admin"]),
        createdAt=user["createdAt"],
        lastLogin=user.get("lastLogin"),
    )


# ── Forgot password ───────────────────────────────────────────────────

@router.post("/forgot-password", response_model=MessageResponse)
def forgot_password(body: ForgotPasswordRequest, db: DatabaseManager = Depends(get_db)):
    from backend.core.email_service import send_reset_email
    
    with db._get_connection() as conn:
        cursor = conn.execute("SELECT * FROM USERS WHERE email = ?", (str(body.email),))
        user = cursor.fetchone()

    if not user:
        # Don't reveal whether email exists
        return MessageResponse(message="If that email is registered, a reset token has been sent.")

    token = create_reset_token(user["userId"], str(body.email))
    
    # Send the email via SMTP
    email_sent = send_reset_email(str(body.email), token)
    
    if not email_sent:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to send reset email. Please try again later."
        )

    return MessageResponse(message="If that email is registered, a reset token has been sent.")


# ── Reset password ────────────────────────────────────────────────────

@router.post("/reset-password", response_model=MessageResponse)
def reset_password(body: ResetPasswordRequest, db: DatabaseManager = Depends(get_db)):
    entry = verify_reset_token(body.token)
    if not entry:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid or expired reset token.")

    db.update_password(entry["user_id"], body.new_password)
    return MessageResponse(message="Password reset successfully. You can now log in.")


# ── Logout ────────────────────────────────────────────────────────────

@router.post("/logout", response_model=MessageResponse)
def logout(_: dict = Depends(get_current_user)):
    """Client-side token deletion. Stateless JWT — server just acknowledges."""
    return MessageResponse(message="Logged out successfully.")
