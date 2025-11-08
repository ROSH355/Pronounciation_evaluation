import json
import re
import pandas as pd
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def _normalize_utt_str(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    try:
        s = Path(s).stem
    except Exception:
        pass
    return s

def _extract_trailing_digits(s: str):
    m = re.search(r'(\d+)$', str(s))
    return m.group(1) if m else None

def _extract_any_digits(s: str):
    parts = re.findall(r'(\d+)', str(s))
    return max(parts, key=len) if parts else None

# ---------------------------------------------------------
# Load score.json and align with features
# ---------------------------------------------------------
def load_scores(score_json_path: Path):
    """Load utterance scores from score.json"""
    with open(score_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            utt = _normalize_utt_str(k)
            if isinstance(v, dict):
                for kk in ("total", "score", "human_score", "final_score", "grade"):
                    if kk in v:
                        rows.append((utt, float(v[kk])))
                        break
            elif isinstance(v, (int, float)):
                rows.append((utt, float(v)))
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            utt = None
            for key in ("utt_id", "id", "file", "filename"):
                if key in item:
                    utt = _normalize_utt_str(item[key])
                    break
            score = None
            for key in ("total", "score", "human_score", "final_score", "grade"):
                if key in item and isinstance(item[key], (int, float)):
                    score = float(item[key])
                    break
            if utt and score is not None:
                rows.append((utt, score))
    df = pd.DataFrame(rows, columns=["utt_id", "human_score"])
    df["utt_id"] = df["utt_id"].astype(str)
    return df.groupby("utt_id", as_index=False)["human_score"].mean()

def load_data(features_csv, score_json):
    print(f"📂 Loading features from {features_csv}")
    features = pd.read_csv(features_csv)

    # 🧩 Fix: ensure utt_id format matches score.json (zero-padded to 9 digits)
    features["utt_id"] = features["utt_id"].astype(str).str.zfill(9)

    with open(score_json, "r", encoding="utf-8") as f:
        scores = json.load(f)

    # Convert score dict to DataFrame
    rows = []
    for utt_id, vals in scores.items():
        if "total" in vals:
            rows.append({"utt_id": utt_id, "score": vals["total"]})
    scores_df = pd.DataFrame(rows)

    # Merge features with scores
    merged = pd.merge(features, scores_df, on="utt_id", how="inner")
    if merged.shape[0] == 0:
        raise ValueError("❌ Could not merge features with scores. Check IDs.")

    X = merged.drop(columns=["file_path", "utt_id", "score"], errors="ignore")
    y = merged["score"]
    print(f"✅ Loaded data — X shape: {X.shape}, y shape: {y.shape}")
    return X, y

# ---------------------------------------------------------
# Train Models
# ---------------------------------------------------------
def train_models(X_train, y_train):
    """Train Gradient Boosting, Random Forest, and SVM"""
    models = {
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),
        "svm": SVR(kernel="rbf")
    }

    fitted = {}
    for name, model in models.items():
        print(f"🚀 Training {name} ...")
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted

# ---------------------------------------------------------
# Save Models
# ---------------------------------------------------------
def save_models(models, models_dir: Path, feature_names=None):
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        payload = {"model": model}
        if feature_names is not None:
            payload["feature_names"] = list(feature_names)
        else:
            fn = getattr(model, "feature_names_in_", None)
            if fn is not None:
                payload["feature_names"] = list(fn)
        joblib.dump(payload, models_dir / f"{name}.pkl")
        print(f"💾 Saved {name} to {models_dir / f'{name}.pkl'}")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / "data/processed"
    models_dir = project_root / "models"
    score_json = project_root / "data/raw/score.json"

    # Load training data
    X_train, y_train = load_data(processed_dir / "features_train_norm.csv", score_json)

    # Train and save
    models = train_models(X_train, y_train)
    save_models(models, models_dir, feature_names=X_train.columns)
