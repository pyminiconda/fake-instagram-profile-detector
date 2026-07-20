"""
schemas/analysis.py — Pydantic models for profile analysis endpoints.
"""

from pydantic import BaseModel
from typing import Optional, Dict, List, Any


# ── Requests ──────────────────────────────────────────────────────────

class LiveAnalysisRequest(BaseModel):
    username: str


class ManualAnalysisRequest(BaseModel):
    username: str = "test_user"
    followers: int = 100
    following: int = 200
    posts: int = 10
    has_pic: bool = True
    bio_length: int = 50
    has_url: bool = False
    full_name: str = ""


# ── Sub-models ────────────────────────────────────────────────────────

class RiskFlag(BaseModel):
    flag: str
    note: str


class ProfileSnapshot(BaseModel):
    username: str
    followersCount: int
    followingCount: int
    postsCount: int
    isPrivate: bool
    isVerified: bool
    hasProfilePicture: bool
    biography: str
    externalUrl: str
    fullName: str


class PredictionResult(BaseModel):
    label: str                          # "fake" | "genuine"
    confidence: float
    low_confidence: bool
    shap_values: Dict[str, float]
    risk_flags: Dict[str, RiskFlag]


# ── Response ──────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    profile: ProfileSnapshot
    prediction: PredictionResult
    history_id: str


class BatchResultItem(BaseModel):
    username: str
    label: str
    confidence: float
    risk_level: str
    timestamp: str
    error: Optional[str] = None


class BatchAnalysisResponse(BaseModel):
    total: int
    fake_count: int
    genuine_count: int
    fake_percentage: float
    results: List[BatchResultItem]
