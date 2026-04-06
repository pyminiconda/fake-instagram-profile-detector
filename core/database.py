"""
DatabaseManager — SQLite operations for the Fake Instagram Profile Detection System.

Handles all database interactions: user management, profile caching,
search history, and model metadata. Uses parameterized queries throughout.
"""

import sqlite3
import uuid
import json
import os
from datetime import datetime, timedelta
from contextlib import contextmanager

import bcrypt


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.db")


class DatabaseManager:
    """Manages all SQLite database operations."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_database()

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------
    @contextmanager
    def _get_connection(self):
        """Context manager for safe database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema initialization
    # ------------------------------------------------------------------
    def _init_database(self):
        """Create all tables and seed default admin on first run."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS USERS (
                    userId      TEXT PRIMARY KEY,
                    username    TEXT UNIQUE NOT NULL,
                    email       TEXT UNIQUE NOT NULL,
                    hashedPassword TEXT NOT NULL,
                    is_admin    INTEGER DEFAULT 0,
                    createdAt   TEXT NOT NULL,
                    lastLogin   TEXT
                );

                CREATE TABLE IF NOT EXISTS PROFILE_CACHE (
                    cacheId           TEXT PRIMARY KEY,
                    username          TEXT NOT NULL,
                    followersCount    INTEGER,
                    followingCount    INTEGER,
                    postsCount        INTEGER,
                    isPrivate         INTEGER,
                    isVerified        INTEGER,
                    hasProfilePicture INTEGER,
                    biography         TEXT,
                    externalUrl       TEXT,
                    fullName          TEXT,
                    fetchedAt         TEXT NOT NULL,
                    expiresAt         TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_cache_username
                    ON PROFILE_CACHE(username);

                CREATE TABLE IF NOT EXISTS SEARCH_HISTORY (
                    historyId        TEXT PRIMARY KEY,
                    userId           TEXT NOT NULL,
                    modelId          TEXT,
                    queriedUsername   TEXT NOT NULL,
                    resultLabel      TEXT NOT NULL,
                    confidenceScore  REAL NOT NULL,
                    featureImportance TEXT,
                    predictedAt      TEXT NOT NULL,
                    exportedAs       TEXT DEFAULT 'none',
                    FOREIGN KEY (userId) REFERENCES USERS(userId)
                );

                CREATE TABLE IF NOT EXISTS MODEL_METADATA (
                    modelId       TEXT PRIMARY KEY,
                    algorithmType TEXT NOT NULL,
                    version       TEXT,
                    accuracy      REAL,
                    precision_score REAL,
                    recall        REAL,
                    f1Score       REAL,
                    aucRoc        REAL,
                    isBestModel   INTEGER DEFAULT 0,
                    filePath      TEXT,
                    trainedAt     TEXT NOT NULL
                );
            """)

            # Seed default admin if USERS table is empty
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM USERS")
            row = cursor.fetchone()
            if row["cnt"] == 0:
                self._seed_admin(conn)

    def _seed_admin(self, conn):
        """Create the default admin account and print a console warning."""
        admin_id = str(uuid.uuid4())
        hashed = bcrypt.hashpw("admin123".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        now = datetime.utcnow().isoformat()
        conn.execute(
            """INSERT INTO USERS (userId, username, email, hashedPassword, is_admin, createdAt)
               VALUES (?, ?, ?, ?, 1, ?)""",
            (admin_id, "admin", "admin@example.com", hashed, now),
        )
        print("=" * 60)
        print("[WARNING] DEFAULT ADMIN ACCOUNT CREATED")
        print("    Username : admin")
        print("    Email    : admin@example.com")
        print("    Password : admin123")
        print("    -> Change the password immediately!")
        print("=" * 60)

    # ------------------------------------------------------------------
    # User operations
    # ------------------------------------------------------------------
    def create_user(self, username: str, email: str, password: str, is_admin: bool = False) -> dict:
        """Register a new user. Returns the created user dict."""
        user_id = str(uuid.uuid4())
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO USERS (userId, username, email, hashedPassword, is_admin, createdAt)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, username, email, hashed, int(is_admin), now),
            )
        return {"userId": user_id, "username": username, "email": email, "is_admin": is_admin}

    def authenticate_user(self, email: str, password: str) -> dict | None:
        """Verify credentials. Returns user dict on success, None on failure."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM USERS WHERE email = ?", (email,))
            user = cursor.fetchone()
            if user and bcrypt.checkpw(password.encode("utf-8"), user["hashedPassword"].encode("utf-8")):
                conn.execute(
                    "UPDATE USERS SET lastLogin = ? WHERE userId = ?",
                    (datetime.utcnow().isoformat(), user["userId"]),
                )
                return dict(user)
        return None

    def get_user_by_id(self, user_id: str) -> dict | None:
        """Fetch a user by userId."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM USERS WHERE userId = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def check_duplicate(self, username: str = None, email: str = None) -> dict:
        """Check if username or email already exists. Returns dict of booleans."""
        result = {"username_exists": False, "email_exists": False}
        with self._get_connection() as conn:
            if username:
                cursor = conn.execute("SELECT 1 FROM USERS WHERE username = ?", (username,))
                result["username_exists"] = cursor.fetchone() is not None
            if email:
                cursor = conn.execute("SELECT 1 FROM USERS WHERE email = ?", (email,))
                result["email_exists"] = cursor.fetchone() is not None
        return result

    def update_password(self, user_id: str, new_password: str):
        """Update a user's password."""
        hashed = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE USERS SET hashedPassword = ? WHERE userId = ?",
                (hashed, user_id),
            )

    # ------------------------------------------------------------------
    # Profile cache operations
    # ------------------------------------------------------------------
    def get_cached_profile(self, username: str) -> dict | None:
        """Return cached profile if it exists and hasn't expired."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM PROFILE_CACHE WHERE username = ? AND expiresAt > ?",
                (username, datetime.utcnow().isoformat()),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def cache_profile(self, profile_data: dict) -> str:
        """Write a profile to cache. Returns the cacheId."""
        cache_id = str(uuid.uuid4())
        now = datetime.utcnow()
        expires = now + timedelta(hours=24)
        with self._get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO PROFILE_CACHE
                   (cacheId, username, followersCount, followingCount, postsCount,
                    isPrivate, isVerified, hasProfilePicture, biography, externalUrl,
                    fullName, fetchedAt, expiresAt)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cache_id,
                    profile_data.get("username", ""),
                    profile_data.get("followersCount", 0),
                    profile_data.get("followingCount", 0),
                    profile_data.get("postsCount", 0),
                    int(profile_data.get("isPrivate", False)),
                    int(profile_data.get("isVerified", False)),
                    int(profile_data.get("hasProfilePicture", False)),
                    profile_data.get("biography", ""),
                    profile_data.get("externalUrl", ""),
                    profile_data.get("fullName", ""),
                    now.isoformat(),
                    expires.isoformat(),
                ),
            )
        return cache_id

    # ------------------------------------------------------------------
    # Search history operations
    # ------------------------------------------------------------------
    def save_search(self, user_id: str, queried_username: str, result_label: str,
                    confidence_score: float, feature_importance: dict = None,
                    model_id: str = None) -> str:
        """Save a search result. Returns historyId."""
        history_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        fi_json = json.dumps(feature_importance) if feature_importance else None
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO SEARCH_HISTORY
                   (historyId, userId, modelId, queriedUsername, resultLabel,
                    confidenceScore, featureImportance, predictedAt)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (history_id, user_id, model_id, queried_username,
                 result_label, confidence_score, fi_json, now),
            )
        return history_id

    def get_history(self, user_id: str, start_date: str = None,
                    end_date: str = None) -> list[dict]:
        """Fetch search history for a user, optionally filtered by date range."""
        query = "SELECT * FROM SEARCH_HISTORY WHERE userId = ?"
        params: list = [user_id]
        if start_date:
            query += " AND predictedAt >= ?"
            params.append(start_date)
        if end_date:
            query += " AND predictedAt <= ?"
            params.append(end_date)
        query += " ORDER BY predictedAt DESC"
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def delete_history(self, history_id: str, user_id: str) -> bool:
        """Delete a search history record. Returns True if deleted."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM SEARCH_HISTORY WHERE historyId = ? AND userId = ?",
                (history_id, user_id),
            )
            return cursor.rowcount > 0

    def update_export_status(self, history_id: str, exported_as: str):
        """Update the exportedAs field for a search record."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE SEARCH_HISTORY SET exportedAs = ? WHERE historyId = ?",
                (exported_as, history_id),
            )

    # ------------------------------------------------------------------
    # Model metadata operations
    # ------------------------------------------------------------------
    def save_model_metadata(self, algorithm_type: str, accuracy: float,
                            precision_score: float, recall: float,
                            f1_score: float, auc_roc: float,
                            is_best: bool, file_path: str,
                            version: str = "1.0") -> str:
        """Save model training metadata. Returns modelId."""
        model_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            # If this is the best model, unset previous best
            if is_best:
                conn.execute("UPDATE MODEL_METADATA SET isBestModel = 0")

            conn.execute(
                """INSERT INTO MODEL_METADATA
                   (modelId, algorithmType, version, accuracy, precision_score,
                    recall, f1Score, aucRoc, isBestModel, filePath, trainedAt)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (model_id, algorithm_type, version, accuracy, precision_score,
                 recall, f1_score, auc_roc, int(is_best), file_path, now),
            )
        return model_id

    def get_best_model(self) -> dict | None:
        """Return metadata for the current best model."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM MODEL_METADATA WHERE isBestModel = 1"
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_models(self) -> list[dict]:
        """Return all model metadata records."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM MODEL_METADATA ORDER BY trainedAt DESC"
            )
            return [dict(r) for r in cursor.fetchall()]
