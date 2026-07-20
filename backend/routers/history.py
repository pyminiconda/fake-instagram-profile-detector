"""
routers/history.py — Search history endpoints.

GET    /api/history          — Get user's history (filterable)
DELETE /api/history/{id}     — Delete one record
GET    /api/history/export/csv  — Export as CSV
GET    /api/history/export/pdf  — Export as PDF
"""

import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import StreamingResponse, PlainTextResponse
from typing import Optional, List
from pydantic import BaseModel

from backend.middleware.auth import get_current_user
from backend.core.database import DatabaseManager
from backend.core.history_manager import HistoryManager

router = APIRouter(prefix="/api/history", tags=["History"])


def get_db() -> DatabaseManager:
    from backend.main import app
    return app.state.db


class HistoryRecord(BaseModel):
    historyId: str
    queriedUsername: str
    resultLabel: str
    confidenceScore: float
    predictedAt: str
    exportedAs: Optional[str] = None


class HistoryListResponse(BaseModel):
    records: List[HistoryRecord]
    total: int


# ── List ──────────────────────────────────────────────────────────────

@router.get("", response_model=HistoryListResponse)
def get_history(
    start_date: Optional[str] = Query(None, description="ISO date string, e.g. 2024-01-01"),
    end_date:   Optional[str] = Query(None, description="ISO date string, e.g. 2024-12-31"),
    current_user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    hm = HistoryManager(db)
    records = hm.get_user_history(
        user_id=current_user["sub"],
        start_date=start_date,
        end_date=end_date,
    )
    return HistoryListResponse(
        records=[HistoryRecord(**r) for r in records],
        total=len(records),
    )


# ── Delete one ────────────────────────────────────────────────────────

@router.delete("/{history_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_history(
    history_id: str,
    current_user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    hm = HistoryManager(db)
    deleted = hm.delete_record(history_id, current_user["sub"])
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Record not found or not owned by you.")


# ── Export CSV ────────────────────────────────────────────────────────

@router.get("/export/csv")
def export_csv(
    current_user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    hm = HistoryManager(db)
    records = hm.get_user_history(current_user["sub"])
    if not records:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="No history records to export.")

    csv_str = hm.export_csv(records)
    return PlainTextResponse(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=history.csv"},
    )


# ── Export PDF ────────────────────────────────────────────────────────

@router.get("/export/pdf")
def export_pdf(
    current_user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    hm = HistoryManager(db)
    records = hm.get_user_history(current_user["sub"])
    if not records:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="No history records to export.")

    user = db.get_user_by_id(current_user["sub"])
    username = user["username"] if user else "User"
    pdf_bytes = hm.export_pdf(records, username)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=history_report.pdf"},
    )
