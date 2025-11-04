# src/models/evaluate.py

import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import json

def load_models(models_dir: Path):
    """Load models saved as dicts with {model, feature_names}"""
    models = {}
    for model_file in models_dir.glob("*.pkl"):
        try:
            payload = joblib.load(model_file)
            if isinstance(payload, dict) and "model" in payload:
                model = payload["model"]
                feature_names = payload.get("feature_names", None)
            else:
                model = payload
                feature_names = getattr(model, "feature_names_in_", None)
            models[model_file.stem] = {"model": model, "feature_names": feature_names}
        except Exception as e:
            print(f"❌ Failed to load {model_file}: {e}")
    print(f"✅ Loaded {len(models)} models from {models_dir}")
    return models

def evaluate_models(models, X, y):
    """Evaluate each model and return results"""
    results = {}
    for name, payload in models.items():
        model = payload["model"]
        feature_names = payload.get("feature_names")

        # Align features if necessary
        X_eval = X.copy()
        if feature_names is not None:
            common = [f for f in feature_names if f in X.columns]
            X_eval = X[common]

        try:
            preds = model.predict(X)
            rmse = mean_squared_error(y, preds) ** 0.5  # FIXED
            r2 = r2_score(y, preds)
            results[name] = {"rmse": rmse, "r2": r2}
            print(f"✅ {name}: RMSE={rmse:.3f}, R²={r2:.3f}")
        except Exception as e:
            print(f"❌ Failed to evaluate {name}: {e}")
            
    return results

def load_test_data(features_csv: Path, score_json: Path, load_data_func):
    """Use the same load_data() from train_model.py"""
    X_test, y_test = load_data_func(features_csv, score_json)
    print(f"✅ Loaded test data: X={X_test.shape}, y={y_test.shape}")
    return X_test, y_test

if __name__ == "__main__":
    from train_model import load_data

    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / "data/processed"
    models_dir = project_root / "models"
    score_json = project_root / "data/raw/score.json"

    test_features_path = processed_dir / "features_test_norm.csv"
    metrics_path = models_dir / "metrics.json"

    # Load test data
    X_test, y_test = load_data(test_features_path, score_json)

    # Load trained models
    models = load_models(models_dir)

    # Evaluate
    results = evaluate_models(models, X_test, y_test)

    # Save metrics
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📊 Saved metrics to {metrics_path}")
