"""
routers/analysis.py — Profile analysis endpoints.

POST /api/analysis/profile   — Live fetch + predict
POST /api/analysis/manual    — Manual feature entry + predict
POST /api/analysis/batch     — Batch CSV upload + predict
GET  /api/analysis/report/{history_id} — Download PDF report
"""

import sys, os, io, csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, status
from fastapi.responses import StreamingResponse

from backend.schemas.analysis import (
    LiveAnalysisRequest, ManualAnalysisRequest,
    AnalysisResponse, BatchAnalysisResponse, BatchResultItem,
    ProfileSnapshot, PredictionResult, RiskFlag,
)
from backend.middleware.auth import get_current_user
from backend.core.database import DatabaseManager
from backend.core.prediction_engine import PredictionEngine
from backend.core.feature_extractor import FeatureExtractor
from backend.core.instaloader_fetch import ProfileFetcher, ProfileNotFoundError, RateLimitError, PrivateProfileError
from backend.core.history_manager import HistoryManager
from backend.core.report_generator import ReportGenerator

router = APIRouter(prefix="/api/analysis", tags=["Analysis"])


# ── Dependency helpers ────────────────────────────────────────────────

def get_db() -> DatabaseManager:
    from backend.main import app
    return app.state.db

def get_engine() -> PredictionEngine:
    from backend.main import app
    return app.state.engine

def get_fetcher() -> ProfileFetcher:
    from backend.main import app
    return app.state.fetcher

def _build_response(profile: dict, prediction: dict, history_id: str) -> AnalysisResponse:
    risk_flags = {
        k: RiskFlag(flag=v["flag"], note=v["note"])
        for k, v in prediction.get("risk_flags", {}).items()
    }
    return AnalysisResponse(
        profile=ProfileSnapshot(**{
            "username": profile.get("username", ""),
            "followersCount": int(profile.get("followersCount", 0)),
            "followingCount": int(profile.get("followingCount", 0)),
            "postsCount": int(profile.get("postsCount", 0)),
            "isPrivate": bool(profile.get("isPrivate", False)),
            "isVerified": bool(profile.get("isVerified", False)),
            "hasProfilePicture": bool(profile.get("hasProfilePicture", False)),
            "biography": profile.get("biography", ""),
            "externalUrl": profile.get("externalUrl", ""),
            "fullName": profile.get("fullName", ""),
        }),
        prediction=PredictionResult(
            label=prediction["label"],
            confidence=prediction["confidence"],
            low_confidence=prediction["low_confidence"],
            shap_values=prediction.get("shap_values", {}),
            risk_flags=risk_flags,
        ),
        history_id=history_id,
    )


# ── Live fetch ────────────────────────────────────────────────────────

@router.post("/profile", response_model=AnalysisResponse)
def analyze_profile(
    body: LiveAnalysisRequest,
    current_user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
    engine: PredictionEngine = Depends(get_engine),
    fetcher: ProfileFetcher = Depends(get_fetcher),
):
    if not engine.is_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Model not loaded. Ask an admin to train the model first.")

    username = body.username.strip().lstrip("@").lower()
    if not username:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username cannot be empty.")

    try:
        profile = fetcher.fetch_profile(username)
    except ProfileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PrivateProfileError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RateLimitError as e:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    extractor = FeatureExtractor()
    features = extractor.extract_from_profile(profile)
    prediction = engine.predict(features)

    hm = HistoryManager(db)
    history_id = hm.save_result(
        user_id=current_user["sub"],
        username=username,
        label=prediction["label"],
        confidence=prediction["confidence"],
        shap_values=prediction.get("shap_values"),
    )

    return _build_response(profile, prediction, history_id)


# ── Manual entry ──────────────────────────────────────────────────────

@router.post("/manual", response_model=AnalysisResponse)
def analyze_manual(
    body: ManualAnalysisRequest,
    current_user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
    engine: PredictionEngine = Depends(get_engine),
):
    if not engine.is_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Model not loaded.")

    extractor = FeatureExtractor()
    features = extractor.extract_from_manual_input(
        followers=body.followers,
        following=body.following,
        posts=body.posts,
        has_pic=body.has_pic,
        bio_length=body.bio_length,
        username=body.username,
        has_url=body.has_url,
        full_name=body.full_name,
    )
    prediction = engine.predict(features)

    profile = {
        "username": body.username,
        "followersCount": body.followers,
        "followingCount": body.following,
        "postsCount": body.posts,
        "isPrivate": False,
        "isVerified": False,
        "hasProfilePicture": body.has_pic,
        "biography": "x" * body.bio_length,
        "externalUrl": "https://example.com" if body.has_url else "",
        "fullName": body.full_name,
    }

    hm = HistoryManager(db)
    history_id = hm.save_result(
        user_id=current_user["sub"],
        username=body.username,
        label=prediction["label"],
        confidence=prediction["confidence"],
        shap_values=prediction.get("shap_values"),
    )

    return _build_response(profile, prediction, history_id)


# ── Batch CSV upload ──────────────────────────────────────────────────

@router.post("/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
    engine: PredictionEngine = Depends(get_engine),
    fetcher: ProfileFetcher = Depends(get_fetcher),
):
    if not engine.is_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Model not loaded.")

    if not file.filename.endswith((".csv", ".txt")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Only CSV files are supported.")

    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))

    usernames = []
    for row in reader:
        if row:
            username = row[0].strip().lstrip("@").lower()
            if username and username != "username":  # skip header
                usernames.append(username)

    if not usernames:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="No usernames found in the CSV file.")

    if len(usernames) > 50:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Batch limit is 50 profiles per request.")

    extractor = FeatureExtractor()
    hm = HistoryManager(db)
    results = []
    now = datetime.utcnow().isoformat()

    for username in usernames:
        try:
            profile = fetcher.fetch_profile(username)
            features = extractor.extract_from_profile(profile)
            prediction = engine.predict(features)

            confidence = prediction["confidence"]
            if confidence >= 0.85:
                risk_level = "High"
            elif confidence >= 0.70:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            hm.save_result(
                user_id=current_user["sub"],
                username=username,
                label=prediction["label"],
                confidence=confidence,
                shap_values=prediction.get("shap_values"),
            )

            results.append(BatchResultItem(
                username=username,
                label=prediction["label"],
                confidence=confidence,
                risk_level=risk_level,
                timestamp=now,
            ))
        except Exception as exc:
            results.append(BatchResultItem(
                username=username,
                label="error",
                confidence=0.0,
                risk_level="Unknown",
                timestamp=now,
                error=str(exc),
            ))

    total = len(results)
    fake_count = sum(1 for r in results if r.label == "fake")
    genuine_count = sum(1 for r in results if r.label == "genuine")

    return BatchAnalysisResponse(
        total=total,
        fake_count=fake_count,
        genuine_count=genuine_count,
        fake_percentage=round(fake_count / max(total, 1) * 100, 1),
        results=results,
    )


# ── PDF Report download ───────────────────────────────────────────────

@router.get("/report/{history_id}")
def download_report(
    history_id: str,
    current_user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db),
):
    records = db.get_history(current_user["sub"])
    record = next((r for r in records if r["historyId"] == history_id), None)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found.")

    gen = ReportGenerator()
    pdf_bytes = gen.generate_history_report([record], record["queriedUsername"])

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report_{history_id[:8]}.pdf"},
    )
