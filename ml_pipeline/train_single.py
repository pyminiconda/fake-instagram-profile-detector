import sys
import os
import joblib

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ml_pipeline.ml.data_manager import DataManager
from ml_pipeline.ml.preprocessor import Preprocessor
from ml_pipeline.ml.model_trainer import ModelTrainer
from backend.core.database import DatabaseManager

def train_single_model(algorithm: str):
    """Train a single algorithm, evaluate it, save it to disk, and record in DB."""
    algo_map = {
        "RandomForest": "RF",
        "SVM": "SVM",
        "LogisticRegression": "LR",
        "XGBoost": "XGB"
    }
    short_name = algo_map.get(algorithm)
    if not short_name:
        raise ValueError(f"Unknown algorithm {algorithm}")
        
    print(f"--- Training {algorithm} ({short_name}) ---")
    
    # 1. Load data
    dm = DataManager()
    df = dm.load_dataset(force_download=False)
    df = dm.clean()
    label_col = dm._find_label_column()
    
    # 2. Preprocess
    prep = Preprocessor()
    prep_data = prep.run_full_pipeline(df, label_col=label_col)
    
    # 3. Train
    trainer = ModelTrainer()
    model_obj = trainer._create_models()[short_name]
    
    X_train = prep_data["X_train"]
    y_train = prep_data["y_train"]
    X_test = prep_data["X_test"]
    y_test = prep_data["y_test"]
    
    model_obj.fit(X_train, y_train)
    
    # 4. Evaluate
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    y_pred = model_obj.predict(X_test)
    if hasattr(model_obj, "predict_proba"):
        y_prob = model_obj.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    # 5. Save model to disk
    models_dir = os.path.join(PROJECT_ROOT, "ml_pipeline", "models")
    os.makedirs(models_dir, exist_ok=True)
    file_path = os.path.join(models_dir, f"{short_name}_latest.joblib")
    joblib.dump(model_obj, file_path)
    
    print(f"Model saved to {file_path}")
    print(f"Metrics - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    # 6. Save to DB
    # We will mark it as the best model (is_best=True), but it won't be loaded into the active engine until manual reload.
    db = DatabaseManager()
    
    # Ensure relative path for DB portability
    rel_path = os.path.join("ml_pipeline", "models", f"{short_name}_latest.joblib")
    
    db.save_model_metadata(
        algorithm_type=algorithm,
        version="1.0",
        accuracy=float(acc),
        precision_score=float(prec),
        recall=float(rec),
        f1_score=float(f1),
        auc_roc=float(auc),
        is_best=True, 
        file_path=rel_path
    )
    print("Saved model metadata to DB.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_single_model(sys.argv[1])
    else:
        print("Usage: python train_single.py <Algorithm>")
