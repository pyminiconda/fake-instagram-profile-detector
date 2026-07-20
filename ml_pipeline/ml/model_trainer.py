"""
ModelTrainer — Train all 4 ML models with Stratified K-Fold cross-validation.

Models: Random Forest, SVM, Logistic Regression, XGBoost.
Includes hyperparameter tuning via RandomizedSearchCV.
"""

import numpy as np
import pandas as pd
from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV


# Default hyperparameter grids for tuning
PARAM_GRIDS = {
    "RF": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    "SVM": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto", 0.01, 0.001],
    },
    "LR": {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [500, 1000, 2000],
    },
    "XGB": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    },
}


class ModelTrainer:
    """Train, cross-validate, and tune ML models."""

    def __init__(self):
        self.models: Dict[str, object] = {}
        self.cv_results: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------
    @staticmethod
    def _create_models() -> Dict[str, object]:
        """Instantiate all 4 models with default parameters."""
        return {
            "RF": RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1
            ),
            "SVM": SVC(
                kernel="rbf", probability=True, random_state=42
            ),
            "LR": LogisticRegression(
                max_iter=1000, random_state=42
            ),
            "XGB": XGBClassifier(
                n_estimators=200, use_label_encoder=False,
                eval_metric="logloss", random_state=42, n_jobs=-1,
            ),
        }

    # ------------------------------------------------------------------
    # Training with cross-validation
    # ------------------------------------------------------------------
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray,
                  k_folds: int = 5,
                  progress_callback=None) -> Dict[str, object]:
        """
        Train all 4 models using Stratified K-Fold cross-validation.

        Args:
            X_train: scaled training features
            y_train: training labels
            k_folds: number of folds for cross-validation
            progress_callback: optional callable(model_name, status_text) for UI

        Returns:
            Dict mapping model name to fitted model object
        """
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        models = self._create_models()

        for i, (name, model) in enumerate(models.items()):
            if progress_callback:
                progress_callback(name, f"Training {name}...")

            print(f"\n[TRAIN] Training {name}...")

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=skf, scoring="f1", n_jobs=-1
            )
            print(f"   CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            # Fit on full training set
            model.fit(X_train, y_train)

            self.models[name] = model
            self.cv_results[name] = {
                "cv_scores": cv_scores,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
            }

            if progress_callback:
                progress_callback(name, f"{name} done — CV F1: {cv_scores.mean():.4f}")

        return self.models

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------
    def tune_best(self, model_name: str,
                  X_train: np.ndarray, y_train: np.ndarray,
                  n_iter: int = 50,
                  k_folds: int = 5) -> tuple:
        """
        Tune the specified model using RandomizedSearchCV.

        Args:
            model_name: one of 'RF', 'SVM', 'LR', 'XGB'
            X_train: training features
            y_train: training labels
            n_iter: number of parameter combinations to try
            k_folds: CV folds

        Returns:
            (best_model, best_params, best_score)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet.")

        print(f"\n[TUNE] Tuning {model_name} with RandomizedSearchCV ({n_iter} iterations)...")

        base_model = self._create_models()[model_name]
        param_grid = PARAM_GRIDS.get(model_name, {})

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            base_model, param_grid,
            n_iter=min(n_iter, self._count_combinations(param_grid)),
            cv=skf, scoring="f1",
            random_state=42, n_jobs=-1, verbose=0,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_

        print(f"   Best params: {best_params}")
        print(f"   Best CV F1:  {best_score:.4f}")

        # Update stored model
        self.models[model_name] = best_model

        return best_model, best_params, best_score

    @staticmethod
    def _count_combinations(param_grid: dict) -> int:
        """Count total parameter combinations."""
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------
    def get_model(self, name: str):
        """Return a trained model by name."""
        return self.models.get(name)

    def get_cv_results(self) -> Dict[str, dict]:
        """Return cross-validation results for all models."""
        return self.cv_results

    def get_all_models(self) -> Dict[str, object]:
        """Return all trained models."""
        return self.models
