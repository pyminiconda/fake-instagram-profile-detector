"""
Preprocessor — Feature engineering, normalization, encoding, SMOTE.

Computes the 7 derived features from raw dataset columns, scales them,
handles class imbalance, and **saves the scaler as models/scaler.pkl**.
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from core.feature_extractor import FeatureExtractor, FEATURE_NAMES

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")


class Preprocessor:
    """Feature engineering, scaling, encoding, and SMOTE."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_extractor = FeatureExtractor()

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def engineer_features(self, df: pd.DataFrame,
                          label_col: str = "is_fake") -> tuple:
        """
        Compute the 7 derived features from raw dataset columns.

        Args:
            df: cleaned DataFrame with raw columns
            label_col: name of the target column

        Returns:
            (X_engineered: DataFrame, y: Series)
        """
        features_list = []
        for _, row in df.iterrows():
            features_list.append(
                self.feature_extractor.extract_from_dataset_row(row)
            )

        X = pd.DataFrame(features_list, columns=FEATURE_NAMES, index=df.index)
        y = df[label_col].astype(int)

        print(f"[FEATURES] Engineered {len(FEATURE_NAMES)} features from {len(df)} samples")
        print(f"   Features: {FEATURE_NAMES}")

        return X, y

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def normalize(self, X_train: pd.DataFrame,
                  X_test: pd.DataFrame) -> tuple:
        """
        Fit StandardScaler on X_train, transform both X_train and X_test.
        Saves the fitted scaler to models/scaler.pkl.

        Returns:
            (X_train_scaled: ndarray, X_test_scaled: ndarray, scaler)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Save scaler for use in PredictionEngine
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self.scaler, SCALER_PATH)
        print(f"[SAVED] Scaler saved to {SCALER_PATH}")

        return X_train_scaled, X_test_scaled, self.scaler

    # ------------------------------------------------------------------
    # Label encoding
    # ------------------------------------------------------------------
    @staticmethod
    def encode_labels(y: pd.Series) -> pd.Series:
        """Ensure labels are binary integers (0/1)."""
        y = y.astype(int)
        unique = sorted(y.unique())
        print(f"[LABELS] Labels: {unique}, counts: {y.value_counts().to_dict()}")
        return y

    # ------------------------------------------------------------------
    # SMOTE
    # ------------------------------------------------------------------
    @staticmethod
    def apply_smote(X_train: np.ndarray,
                    y_train: pd.Series,
                    random_state: int = 42) -> tuple:
        """
        Apply SMOTE to balance the training set.

        Returns:
            (X_resampled, y_resampled)
        """
        print(f"[SMOTE] Before SMOTE: {pd.Series(y_train).value_counts().to_dict()}")

        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        print(f"[SMOTE] After  SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
        return X_res, y_res

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run_full_pipeline(self, df: pd.DataFrame,
                          label_col: str = "is_fake",
                          test_size: float = 0.2,
                          apply_smote: bool = True) -> dict:
        """
        Run the complete preprocessing pipeline:
          1. Engineer features
          2. Encode labels
          3. Train/test split
          4. Normalize (and save scaler)
          5. Apply SMOTE (optional)

        Returns:
            dict with X_train, X_test, y_train, y_test, scaler, feature_names
        """
        from sklearn.model_selection import train_test_split

        # 1. Engineer features
        X, y = self.engineer_features(df, label_col)

        # 2. Encode labels
        y = self.encode_labels(y)

        # 3. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"[SPLIT] train={len(X_train)}, test={len(X_test)}")

        # 4. Normalize (saves scaler)
        X_train_scaled, X_test_scaled, scaler = self.normalize(X_train, X_test)

        # 5. SMOTE
        if apply_smote:
            X_train_scaled, y_train = self.apply_smote(X_train_scaled, y_train)

        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": np.array(y_train),
            "y_test": np.array(y_test),
            "scaler": scaler,
            "feature_names": FEATURE_NAMES,
            "X_train_df": X_train,   # pre-scaling DataFrame (for inspection)
            "X_test_df": X_test,
        }
