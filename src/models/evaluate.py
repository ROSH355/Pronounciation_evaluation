# src/models/evaluate.py

import argparse
import json
from math import sqrt
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_load_models(models_dir: Path):
    """Load all .pkl models safely."""
    models = {}
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    for p in sorted(models_dir.glob("*.pkl")):
        try:
            models[p.stem] = joblib.load(p)
        except Exception as e:
            print(f"⚠ Failed to load model {p.name}: {e}")
    if not models:
        raise ValueError(f"No models found in {models_dir}")
    return models


def _align_features_for_model(X: pd.DataFrame, model):
    """Ensure feature order matches model expectation."""
    fn = getattr(model, "feature_names_in_", None)
    if fn is None:
        return X
    fn = list(fn)
    Xc = X.copy()
    for col in fn:
        if col not in Xc.columns:
            Xc[col] = 0.0
    extras = [c for c in Xc.columns if c not in fn]
    if extras:
        Xc = Xc.drop(columns=extras)
    return Xc[fn]


def evaluate_model_on_data(model, X: pd.DataFrame, y: pd.Series):
    X_aligned = _align_features_for_model(X, model)
    preds = model.predict(X_aligned)
    mae = mean_absolute_error(y, preds)
    rmse = sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def load_data_from_train_module(features_csv: Path, score_json: Path):
    """Reuse load_data() logic from train_model.py."""
    try:
        from src.models.train_model import load_data
    except ImportError:
        from train_model import load_data
    return load_data(features_csv, score_json)


def main():
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Evaluate saved models on test set")
    parser.add_argument("--features", type=Path, default=project_root / "data/processed/features_test_norm.csv")
    parser.add_argument("--scores", type=Path, default=project_root / "data/raw/score.json")
    parser.add_argument("--models-dir", type=Path, default=project_root / "models")
    parser.add_argument("--output", type=Path, default=project_root / "models/metrics.json")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names to evaluate (without .pkl)")
    args = parser.parse_args()

    # Load test data
    X, y = load_data_from_train_module(args.features, args.scores)
    print(f"✅ Loaded test data: X={X.shape}, y={y.shape}")

    # Load models
    models = _safe_load_models(args.models_dir)
    if args.models:
        requested = {m.strip() for m in args.models.split(",")}
        models = {n: m for n, m in models.items() if n in requested}
        if not models:
            raise ValueError(f"No requested models found: {requested}")

    results = {}
    for name, model in models.items():
        try:
            metrics = evaluate_model_on_data(model, X, y)
            results[name] = metrics
            print(f"{name}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
        except Exception as e:
            print(f"❌ Failed to evaluate {name}: {e}")

    # Save metrics JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"📊 Saved metrics to {args.output}")


if __name__ == "__main__":
    main()