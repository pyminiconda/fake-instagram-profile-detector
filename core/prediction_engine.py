"""
PredictionEngine — Loads the trained model + scaler and runs inference.

Loads both:
  • models/best_model.pkl  — the trained classifier
  • models/scaler.pkl      — the StandardScaler used during training

Pipeline: raw features → scaler.transform() → model.predict() / predict_proba()
Also computes SHAP values for explainability.
"""

import os
import numpy as np
import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

FEATURE_NAMES = [
    "followerRatio",
    "profileCompleteness",
    "engagementRate",
    "bioLength",
    "usernameAnomalyScore",
    "postFrequency",
    "hasPicture",
]

CONFIDENCE_THRESHOLD = 0.70


class PredictionEngine:
    """Load model + scaler and run predictions with SHAP explainability."""

    def __init__(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        self.model = None
        self.scaler = None
        self.explainer = None
        self._model_path = model_path
        self._scaler_path = scaler_path
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self):
        """Load model and scaler from disk."""
        if os.path.exists(self._model_path):
            self.model = joblib.load(self._model_path)
        else:
            print(f"[WARNING] Model file not found at {self._model_path}")
            print("   Train a model first via the Admin -> Model Training page.")

        if os.path.exists(self._scaler_path):
            self.scaler = joblib.load(self._scaler_path)
        else:
            print(f"[WARNING] Scaler file not found at {self._scaler_path}")

    def is_ready(self) -> bool:
        """Return True if both model and scaler are loaded."""
        return self.model is not None and self.scaler is not None

    def reload(self):
        """Reload model and scaler from disk (after retraining)."""
        self._load()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, raw_features: list) -> dict:
        """
        Run prediction on a raw (un-scaled) feature vector.

        Args:
            raw_features: list of 7 floats in FEATURE_NAMES order

        Returns:
            dict with keys:
                label        — 'fake' or 'genuine'
                confidence   — float 0-1
                low_confidence — bool (True if confidence < threshold)
                shap_values  — dict mapping feature name to SHAP value
                risk_flags   — dict mapping feature name to risk assessment
        """
        if not self.is_ready():
            raise RuntimeError("Model or scaler not loaded. Train the model first.")

        X = np.array(raw_features).reshape(1, -1)

        # Scale features using the training scaler
        X_scaled = self.scaler.transform(X)

        # Predict
        label_int = int(self.model.predict(X_scaled)[0])
        label = "fake" if label_int == 1 else "genuine"

        # Confidence
        probas = self.model.predict_proba(X_scaled)[0]
        confidence = float(probas[label_int])
        low_confidence = confidence < CONFIDENCE_THRESHOLD

        # SHAP explainability
        shap_values = self._compute_shap(X_scaled)

        # Risk flags based on raw feature values
        risk_flags = self._assess_risk(raw_features)

        return {
            "label": label,
            "confidence": confidence,
            "low_confidence": low_confidence,
            "shap_values": shap_values,
            "risk_flags": risk_flags,
        }

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------
    def _compute_shap(self, X_scaled: np.ndarray) -> dict:
        """Compute SHAP values for a single prediction."""
        try:
            import shap

            if self.explainer is None:
                # Use TreeExplainer for tree-based models, otherwise KernelExplainer
                model_type = type(self.model).__name__
                if model_type in ("RandomForestClassifier", "XGBClassifier",
                                  "GradientBoostingClassifier"):
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    # Fallback — KernelExplainer is slower but universal
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba, X_scaled
                    )

            sv = self.explainer.shap_values(X_scaled)

            # TreeExplainer may return a list (one per class) — take class 1 (fake)
            if isinstance(sv, list):
                values = sv[1][0] if len(sv) > 1 else sv[0][0]
            else:
                # For binary, take positive class column if 2D
                if sv.ndim == 3:
                    values = sv[0, :, 1]
                else:
                    values = sv[0]

            return {name: float(v) for name, v in zip(FEATURE_NAMES, values)}

        except Exception as exc:
            print(f"[WARNING] SHAP computation failed: {exc}")
            return {name: 0.0 for name in FEATURE_NAMES}

    # ------------------------------------------------------------------
    # Risk assessment
    # ------------------------------------------------------------------
    @staticmethod
    def _assess_risk(raw_features: list) -> dict:
        """
        Generate per-feature risk flags from raw (un-scaled) values.

        Returns dict: feature_name → { 'flag': '⚠️' or '✅', 'note': str }
        """
        (follower_ratio, completeness, engagement, bio_len,
         anomaly, posts, has_pic) = raw_features

        flags = {}

        # Follower ratio
        if follower_ratio < 0.1:
            flags["followerRatio"] = {"flag": "⚠️", "note": "Very low follower-to-following ratio"}
        elif follower_ratio < 0.5:
            flags["followerRatio"] = {"flag": "⚠️", "note": "Low follower-to-following ratio"}
        else:
            flags["followerRatio"] = {"flag": "✅", "note": "Healthy follower ratio"}

        # Profile completeness
        if completeness < 0.4:
            flags["profileCompleteness"] = {"flag": "⚠️", "note": "Incomplete profile"}
        else:
            flags["profileCompleteness"] = {"flag": "✅", "note": "Profile is well filled out"}

        # Engagement
        if engagement > 5.0:
            flags["engagementRate"] = {"flag": "⚠️", "note": "Unusually high post-to-follower ratio"}
        elif engagement < 0.001 and posts > 0:
            flags["engagementRate"] = {"flag": "⚠️", "note": "Very low engagement"}
        else:
            flags["engagementRate"] = {"flag": "✅", "note": "Normal engagement"}

        # Bio length
        if bio_len == 0:
            flags["bioLength"] = {"flag": "⚠️", "note": "No biography"}
        else:
            flags["bioLength"] = {"flag": "✅", "note": "Biography present"}

        # Username anomaly
        if anomaly > 0.3:
            flags["usernameAnomalyScore"] = {"flag": "⚠️", "note": "Username looks auto-generated"}
        else:
            flags["usernameAnomalyScore"] = {"flag": "✅", "note": "Username looks organic"}

        # Post frequency
        if posts == 0:
            flags["postFrequency"] = {"flag": "⚠️", "note": "No posts"}
        elif posts < 3:
            flags["postFrequency"] = {"flag": "⚠️", "note": "Very few posts"}
        else:
            flags["postFrequency"] = {"flag": "✅", "note": "Active posting history"}

        # Profile picture
        if has_pic < 0.5:
            flags["hasPicture"] = {"flag": "⚠️", "note": "No profile picture"}
        else:
            flags["hasPicture"] = {"flag": "✅", "note": "Profile picture present"}

        return flags
