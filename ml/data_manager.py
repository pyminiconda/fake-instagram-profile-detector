"""
DataManager — Load, explore, clean, and split the InstaFake dataset.

Auto-downloads from GitHub on first use and caches locally at data/instafake_dataset.csv.
"""

import os
import urllib.request

import pandas as pd
import numpy as np

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOCAL_CSV = os.path.join(DATA_DIR, "instafake_dataset.csv")

# GitHub raw URL for the InstaFake dataset (fake account detection set)
DATASET_URL = (
    "https://raw.githubusercontent.com/fcakyon/instafake-dataset/"
    "master/data/fake-account-detection/insta_fake.csv"
)


class DataManager:
    """Load, explore, clean, and split the InstaFake dataset."""

    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.df_clean: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_dataset(self, force_download: bool = False) -> pd.DataFrame:
        """
        Load the InstaFake CSV. Downloads from GitHub if not cached locally.

        Returns:
            Raw DataFrame
        """
        os.makedirs(DATA_DIR, exist_ok=True)

        if not os.path.exists(LOCAL_CSV) or force_download:
            print(f"[INFO] Downloading dataset from GitHub...")
            try:
                urllib.request.urlretrieve(DATASET_URL, LOCAL_CSV)
                print(f"[OK] Saved to {LOCAL_CSV}")
            except Exception as exc:
                # Try alternative URL patterns
                alt_urls = [
                    "https://raw.githubusercontent.com/fcakyon/instafake-dataset/master/data/insta_fake.csv",
                    "https://raw.githubusercontent.com/fcakyon/instafake-dataset/main/data/fake-account-detection/insta_fake.csv",
                    "https://raw.githubusercontent.com/fcakyon/instafake-dataset/main/data/insta_fake.csv",
                ]
                downloaded = False
                for url in alt_urls:
                    try:
                        urllib.request.urlretrieve(url, LOCAL_CSV)
                        print(f"[OK] Saved to {LOCAL_CSV}")
                        downloaded = True
                        break
                    except Exception:
                        continue
                if not downloaded:
                    raise FileNotFoundError(
                        f"Could not download dataset. Please download manually from "
                        f"https://github.com/fcakyon/instafake-dataset and place the "
                        f"CSV at {LOCAL_CSV}\nOriginal error: {exc}"
                    )

        self.df = pd.read_csv(LOCAL_CSV)
        print(f"[INFO] Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    # ------------------------------------------------------------------
    # Exploration helpers
    # ------------------------------------------------------------------
    def get_info(self) -> dict:
        """Return dataset summary statistics."""
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        label_col = self._find_label_column()
        label_counts = self.df[label_col].value_counts().to_dict()

        return {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "column_names": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "label_distribution": label_counts,
            "label_column": label_col,
        }

    def _find_label_column(self) -> str:
        """Find the label/target column in the dataset."""
        candidates = ["is_fake", "label", "fake", "is_bot", "automated_behaviour"]
        for col in candidates:
            if col in self.df.columns:
                return col
        # Default: last column
        return self.df.columns[-1]

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    def clean(self) -> pd.DataFrame:
        """
        Clean the raw dataset:
          1. Remove duplicates
          2. Handle missing values (median imputation for numeric, mode for categorical)

        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        df = self.df.copy()
        initial_rows = len(df)

        # 1. Remove duplicates
        df = df.drop_duplicates()
        dupes_removed = initial_rows - len(df)
        print(f"[CLEAN] Removed {dupes_removed} duplicate rows ({len(df)} remaining)")

        # 2. Handle missing values
        missing_before = df.isnull().sum().sum()
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ["int64", "float64"]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        missing_after = df.isnull().sum().sum()
        print(f"[FIX] Missing values: {missing_before} -> {missing_after}")

        self.df_clean = df
        return df

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
    def split(self, test_size: float = 0.2,
              random_state: int = 42) -> tuple:
        """
        Stratified train/test split.

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split

        df = self.df_clean if self.df_clean is not None else self.df
        if df is None:
            raise ValueError("No data available. Load and clean first.")

        label_col = self._find_label_column()
        X = df.drop(columns=[label_col])
        y = df[label_col]

        # Remove non-numeric columns for ML
        X = X.select_dtypes(include=[np.number])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y,
        )

        print(f"[SPLIT] train={len(X_train)}, test={len(X_test)}")
        print(f"   Train class dist: {y_train.value_counts().to_dict()}")
        print(f"   Test  class dist: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test
