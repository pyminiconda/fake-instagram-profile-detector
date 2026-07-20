"""
main.py — FastAPI application entry point for InstaGuard.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.config import APP_NAME, APP_VERSION, FRONTEND_ORIGIN
from backend.routers import auth, analysis, history, users, admin
from backend.core.database import DatabaseManager
from backend.core.prediction_engine import PredictionEngine
from backend.core.instaloader_fetch import ProfileFetcher


# ── Lifespan: boot shared services once ──────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"\n{'='*55}")
    print(f"  {APP_NAME} API v{APP_VERSION} — Starting up...")
    print(f"{'='*55}")

    app.state.db      = DatabaseManager()
    app.state.engine  = PredictionEngine()
    app.state.fetcher = ProfileFetcher(app.state.db)

    print(f"  [DB]     SQLite connected.")
    print(f"  [ML]     Model ready: {app.state.engine.is_ready()}")
    print(f"  [Fetch]  Demo mode: {app.state.fetcher.is_demo_mode()}")
    print(f"  [Docs]   http://localhost:8000/docs")
    print(f"{'='*55}\n")

    yield  # app is running

    print("\n[Shutdown] InstaGuard API stopped.")


# ── App factory ───────────────────────────────────────────────────────

app = FastAPI(
    title=f"{APP_NAME} API",
    description="AI-powered Instagram fake profile detection — REST API",
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(analysis.router)
app.include_router(history.router)
app.include_router(users.router)
app.include_router(admin.router)


# ── Health check ──────────────────────────────────────────────────────
@app.get("/api/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION,
    }


# ── Dev entrypoint ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
