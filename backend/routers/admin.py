"""
routers/admin.py — Admin-only endpoints.

GET  /api/admin/stats                   — System statistics
GET  /api/admin/users                   — List all users
POST /api/admin/users                   — Create a user
GET  /api/admin/users/{id}              — Get user detail
DELETE /api/admin/users/{id}            — Delete a user
PUT  /api/admin/users/{id}/role         — Promote/demote
PUT  /api/admin/users/{id}/password     — Reset user's password
GET  /api/admin/dataset/insights        — Dataset analytics
POST /api/admin/model/train             — Trigger training
GET  /api/admin/model/status            — Model metrics
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from typing import List

from backend.middleware.auth import get_current_admin
from backend.schemas.users import (
    CreateUserRequest, UpdateRoleRequest, AdminResetPasswordRequest,
    UserDetailResponse, UserListResponse,
)
from backend.core.database import DatabaseManager

router = APIRouter(prefix="/api/admin", tags=["Admin"])


def get_db() -> DatabaseManager:
    from backend.main import app
    return app.state.db

def get_engine():
    from backend.main import app
    return app.state.engine


# ── System stats ──────────────────────────────────────────────────────

@router.get("/stats")
def get_stats(
    _: dict = Depends(get_current_admin),
    db: DatabaseManager = Depends(get_db),
    engine=Depends(get_engine),
):
    with db._get_connection() as conn:
        users_row    = conn.execute("SELECT COUNT(*) as c FROM USERS").fetchone()
        history_row  = conn.execute("SELECT COUNT(*) as c FROM SEARCH_HISTORY").fetchone()
        fake_row     = conn.execute("SELECT COUNT(*) as c FROM SEARCH_HISTORY WHERE resultLabel='fake'").fetchone()
        genuine_row  = conn.execute("SELECT COUNT(*) as c FROM SEARCH_HISTORY WHERE resultLabel='genuine'").fetchone()
        today        = datetime.utcnow().date().isoformat()
        today_row    = conn.execute(
            "SELECT COUNT(*) as c FROM SEARCH_HISTORY WHERE predictedAt >= ?", (today,)
        ).fetchone()

    total_scans = history_row["c"]
    best_model  = db.get_best_model()

    return {
        "total_users": users_row["c"],
        "total_scans": total_scans,
        "scans_today": today_row["c"],
        "fake_count": fake_row["c"],
        "genuine_count": genuine_row["c"],
        "fake_percentage": round(fake_row["c"] / max(total_scans, 1) * 100, 1),
        "model_ready": engine.is_ready(),
        "model_accuracy": best_model["accuracy"] if best_model else None,
        "model_algorithm": best_model["algorithmType"] if best_model else None,
    }


# ── Users list ────────────────────────────────────────────────────────

@router.get("/users", response_model=UserListResponse)
def list_users(
    _: dict = Depends(get_current_admin),
    db: DatabaseManager = Depends(get_db),
):
    with db._get_connection() as conn:
        rows = conn.execute(
            "SELECT userId, username, email, is_admin, createdAt, lastLogin FROM USERS ORDER BY createdAt DESC"
        ).fetchall()

    users = [
        UserDetailResponse(
            userId=r["userId"],
            username=r["username"],
            email=r["email"],
            is_admin=bool(r["is_admin"]),
            createdAt=r["createdAt"],
            lastLogin=r["lastLogin"],
        )
        for r in rows
    ]
    return UserListResponse(users=users, total=len(users))


# ── Create user ───────────────────────────────────────────────────────

@router.post("/users", response_model=UserDetailResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    body: CreateUserRequest,
    _: dict = Depends(get_current_admin),
    db: DatabaseManager = Depends(get_db),
):
    dupes = db.check_duplicate(username=body.username, email=str(body.email))
    errors = []
    if dupes["username_exists"]:
        errors.append("Username already taken.")
    if dupes["email_exists"]:
        errors.append("Email already registered.")
    if errors:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=" ".join(errors))

    new_user = db.create_user(body.username, str(body.email), body.password, is_admin=body.is_admin)
    full = db.get_user_by_id(new_user["userId"])
    return UserDetailResponse(
        userId=full["userId"],
        username=full["username"],
        email=full["email"],
        is_admin=bool(full["is_admin"]),
        createdAt=full["createdAt"],
        lastLogin=full.get("lastLogin"),
    )


# ── Get user ──────────────────────────────────────────────────────────

@router.get("/users/{user_id}", response_model=UserDetailResponse)
def get_user(
    user_id: str,
    _: dict = Depends(get_current_admin),
    db: DatabaseManager = Depends(get_db),
):
    user = db.get_user_by_id(user_id)
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


# ── Delete user ───────────────────────────────────────────────────────

@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: str,
    current_admin: dict = Depends(get_current_admin),
    db: DatabaseManager = Depends(get_db),
):
    if user_id == current_admin["sub"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="You cannot delete your own admin account.")

    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    if user.get("is_superadmin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="This account is protected and cannot be deleted.")

    with db._get_connection() as conn:
        conn.execute("DELETE FROM SEARCH_HISTORY WHERE userId = ?", (user_id,))
        conn.execute("DELETE FROM USERS WHERE userId = ?", (user_id,))


# ── Update role ───────────────────────────────────────────────────────

@router.put("/users/{user_id}/role", response_model=UserDetailResponse)
def update_role(
    user_id: str,
    body: UpdateRoleRequest,
    current_admin: dict = Depends(get_current_admin),
    db: DatabaseManager = Depends(get_db),
):
    if user_id == current_admin["sub"] and not body.is_admin:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="You cannot demote your own admin account.")

    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    if user.get("is_superadmin") and not body.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="This account is a superadmin and cannot be demoted.")

    with db._get_connection() as conn:
        conn.execute(
            "UPDATE USERS SET is_admin = ? WHERE userId = ?",
            (int(body.is_admin), user_id),
        )

    updated = db.get_user_by_id(user_id)
    return UserDetailResponse(
        userId=updated["userId"],
        username=updated["username"],
        email=updated["email"],
        is_admin=bool(updated["is_admin"]),
        createdAt=updated["createdAt"],
        lastLogin=updated.get("lastLogin"),
    )


# ── Admin reset user password ─────────────────────────────────────────

@router.put("/users/{user_id}/password")
def reset_user_password(
    user_id: str,
    body: AdminResetPasswordRequest,
    _: dict = Depends(get_current_admin),
    db: DatabaseManager = Depends(get_db),
):
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    db.update_password(user_id, body.new_password)
    return {"message": "User password updated successfully."}


# ── User history (admin view) ─────────────────────────────────────────

@router.get("/users/{user_id}/history")
def get_user_history(
    user_id: str,
    _: dict = Depends(get_current_admin),
    db: DatabaseManager = Depends(get_db),
):
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    records = db.get_history(user_id)
    return {"records": records, "total": len(records)}


# ── Dataset insights ──────────────────────────────────────────────────

@router.get("/dataset/insights")
def dataset_insights(_: dict = Depends(get_current_admin)):
    try:
        import pandas as pd
        from backend.config import DATA_DIR

        csv_path = os.path.join(DATA_DIR, "instafake_dataset.csv")
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="Dataset not found.")

        df = pd.read_csv(csv_path)
        total = len(df)

        fake_col = None
        for col in ["fake", "is_fake", "label", "class"]:
            if col in df.columns:
                fake_col = col
                break

        fake_count    = int(df[fake_col].sum()) if fake_col else 0
        genuine_count = total - fake_count if fake_col else total

        numeric_stats = {}
        for col in df.select_dtypes(include=["number"]).columns[:10]:
            numeric_stats[col] = {
                "mean":   round(float(df[col].mean()), 4),
                "std":    round(float(df[col].std()),  4),
                "min":    round(float(df[col].min()),  4),
                "max":    round(float(df[col].max()),  4),
            }

        return {
            "total_records": total,
            "fake_count":    fake_count,
            "genuine_count": genuine_count,
            "columns":       list(df.columns),
            "numeric_stats": numeric_stats,
        }
    except ImportError:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="pandas is required for dataset insights.")


# ── Model training ────────────────────────────────────────────────────

@router.post("/model/train")
def train_model(
    background_tasks: BackgroundTasks,
    algorithm: str = "RandomForest",
    _: dict = Depends(get_current_admin),
    db: DatabaseManager = Depends(get_db),
):
    valid_algorithms = ["RandomForest", "XGBoost", "LogisticRegression", "GradientBoosting", "SVM"]
    if algorithm not in valid_algorithms:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Algorithm must be one of: {valid_algorithms}")

        try:
            from ml_pipeline.train_single import train_single_model
            train_single_model(algorithm)
        except Exception as e:
            print(f"[Training Error] {e}")

    background_tasks.add_task(_train)
    return {"message": f"Training started for {algorithm}. Check /api/admin/model/status for progress."}


# ── Model status ──────────────────────────────────────────────────────

@router.get("/model/status")
def model_status(
    _: dict = Depends(get_current_admin),
    db: DatabaseManager = Depends(get_db),
    engine=Depends(get_engine),
):
    best  = db.get_best_model()
    all_m = db.get_all_models()
    return {
        "model_ready": engine.is_ready(),
        "best_model":  best,
        "all_models":  all_m,
    }

# ── Model reload ──────────────────────────────────────────────────────

@router.post("/model/reload")
def reload_model(
    _: dict = Depends(get_current_admin),
    engine=Depends(get_engine),
):
    try:
        engine.reload()
        return {"message": "Inference engine reloaded successfully with the latest best model."}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
