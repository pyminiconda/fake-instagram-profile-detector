"""
config.py — Application settings loaded from environment variables.
"""

import os
import secrets
from dotenv import load_dotenv

load_dotenv()

# ── JWT ──────────────────────────────────────────────────────────────
SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# ── App ───────────────────────────────────────────────────────────────
APP_NAME: str = "InstaGuard"
APP_VERSION: str = "1.0.0"
FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

# ── Email (SMTP) ─────────────────────────────────────────────────────
SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
SMTP_SENDER: str = os.getenv("SMTP_SENDER", "InstaGuard Security <no-reply@example.com>")

# ── Database ─────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH: str = os.path.join(PROJECT_ROOT, "app.db")

# ── Models ───────────────────────────────────────────────────────────
MODELS_DIR: str = os.path.join(PROJECT_ROOT, "ml_pipeline", "models")
DATA_DIR: str = os.path.join(PROJECT_ROOT, "ml_pipeline", "data")
