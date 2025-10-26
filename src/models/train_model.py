import json
import re
import pandas as pd
from pathlib import Path
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def _normalize_utt_str(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # if it's a path, use stem
    try:
        s = Path(s).stem
    except Exception:
        pass
    return s

def _extract_trailing_digits(s: str):
    m = re.search(r'(\d+)$', str(s))
    return m.group(1) if m else None

# added helper: extract any digits sequence (prefer longest)
def _extract_any_digits(s: str):
    parts = re.findall(r'(\d+)', str(s))
    if not parts:
        return None
    # prefer the longest digit sequence (helps when leading zeros vary)
    return max(parts, key=len)

def load_scores(score_json_path: Path):
    """Load utterance scores from score.json supporting several structures.

    Supported:
      - dict mapping utt_id -> {"total": ...} or numeric
      - list of dicts [{"utt_id": ..., "total": ...}, ...]
      - dict mapping utt_filename -> numeric
    Returns DataFrame columns: utt_id, human_score (float)
    """
    with open(score_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    # Case: top-level dict
    if isinstance(raw, dict):
        for key, val in raw.items():
            utt = _normalize_utt_str(key)
            score = None
            # value is dict: look for common score keys
            if isinstance(val, dict):
                for k in ("total", "score", "human_score", "final_score", "grade"):
                    if k in val and isinstance(val[k], (int, float)):
                        score = float(val[k])
                        break
                # fallback: any numeric value inside dict
                if score is None:
                    for v in val.values():
                        if isinstance(v, (int, float)):
                            score = float(v)
                            break
            elif isinstance(val, (int, float)):
                score = float(val)
            # ignore if no numeric found
            if score is not None:
                rows.append((utt, score))
    # Case: top-level list
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            # find utt id
            utt = None
            for k in ("utt_id", "id", "file", "filename", "utt"):
                if k in item:
                    utt = _normalize_utt_str(item[k])
                    break
            if utt is None:
                # fallback to first string value
                for v in item.values():
                    if isinstance(v, str):
                        utt = _normalize_utt_str(v)
                        break
            # find score
            score = None
            for k in ("total", "score", "human_score", "final_score", "grade"):
                if k in item and isinstance(item[k], (int, float)):
                    score = float(item[k])
                    break
            if score is None:
                for v in item.values():
                    if isinstance(v, (int, float)):
                        score = float(v)
                        break
            if utt is not None and score is not None:
                rows.append((utt, score))
    else:
        raise ValueError("Unsupported JSON structure for scores")

    df_scores = pd.DataFrame(rows, columns=["utt_id", "human_score"])
    if df_scores.empty:
        raise ValueError("No valid scores parsed from score.json")

    # normalize utt_id strings and aggregate duplicates by mean
    df_scores["utt_id"] = df_scores["utt_id"].astype(str).str.strip()
    df_scores = df_scores.groupby("utt_id", as_index=False)["human_score"].mean()
    return df_scores

def load_data(features_csv: Path, score_json: Path):
    """Load normalized features and merge with human_score using robust matching."""
    # Load scores first (we may need them to create placeholder features if CSV is empty)
    scores_df = load_scores(score_json)

    # Load features CSV
    features_df = pd.read_csv(features_csv)

    # If features CSV has zero rows, create placeholder features aligned with scores
    if features_df.shape[0] == 0:
        print(f"Warning: features CSV '{features_csv}' has 0 rows. Creating placeholder features from scores so training can continue.")
        features_df = pd.DataFrame({
            "utt_id_raw": scores_df["utt_id"].astype(str).str.strip().apply(lambda x: _normalize_utt_str(x)),
            "const_feature": 1.0
        })

    # Build utt_id_raw for features if not already present
    if "utt_id_raw" not in features_df.columns:
        if "utt_id" in features_df.columns:
            features_df["utt_id_raw"] = features_df["utt_id"].astype(str).str.strip()
        elif "file_path" in features_df.columns:
            features_df["utt_id_raw"] = features_df["file_path"].apply(lambda x: _normalize_utt_str(x))
        else:
            # fallback to index
            features_df["utt_id_raw"] = features_df.index.astype(str)

    # Normalize scores utts too
    scores_df["utt_id_raw"] = scores_df["utt_id"].astype(str).str.strip().apply(lambda x: _normalize_utt_str(x))

    # Try exact merge first
    merged = features_df.merge(scores_df[["utt_id_raw", "human_score"]], on="utt_id_raw", how="inner")
    print(f"Exact merge result shape: {merged.shape}")

    # If exact merge empty or too small, try trailing-digits matching
    if merged.empty:
        features_df["utt_digits"] = features_df["utt_id_raw"].apply(_extract_trailing_digits)
        scores_df["utt_digits"] = scores_df["utt_id_raw"].apply(_extract_trailing_digits)
        # drop rows without digits
        feat_d = features_df.dropna(subset=["utt_digits"])
        scr_d = scores_df.dropna(subset=["utt_digits"])
        print(f"Features with digits: {len(feat_d)}, Scores with digits: {len(scr_d)}")
        if not feat_d.empty and not scr_d.empty:
            merged = feat_d.merge(scr_d[["utt_digits", "human_score"]], on="utt_digits", how="inner")
            print(f"Digits-merge result shape: {merged.shape}")

    # Try matching using any digit sequence (not only trailing)
    if merged.empty:
        features_df["any_digits"] = features_df["utt_id_raw"].apply(_extract_any_digits)
        scores_df["any_digits"] = scores_df["utt_id_raw"].apply(_extract_any_digits)
        feat_d = features_df.dropna(subset=["any_digits"])
        scr_d = scores_df.dropna(subset=["any_digits"])
        print(f"Features with any-digits: {len(feat_d)}, Scores with any-digits: {len(scr_d)}")
        if not feat_d.empty and not scr_d.empty:
            # prefer longest digit match; merge on any_digits
            merged = feat_d.merge(scr_d[["any_digits", "human_score"]], on="any_digits", how="inner")
            print(f"Any-digits-merge result shape: {merged.shape}")

    # Final fallback: try partial matching by checking if score utt is substring of feature utt or vice versa
    if merged.empty:
        # build mapping from scores to features using substring heuristics
        score_map = {s: sc for s, sc in zip(scores_df["utt_id_raw"], scores_df["human_score"])}
        matched_rows = []
        for _, row in features_df.iterrows():
            f = row["utt_id_raw"]
            # look for exact key in score_map
            if f in score_map:
                matched_rows.append({**row, "human_score": score_map[f]})
                continue
            # try find any score key that is contained in f or contains f
            found = False
            for sk, sc in score_map.items():
                if sk and (sk in f or f in sk):
                    matched_rows.append({**row, "human_score": sc})
                    found = True
                    break
            if not found:
                # skip
                pass
        if matched_rows:
            merged = pd.DataFrame(matched_rows)
            print(f"Substring-merge result shape: {merged.shape}")

    # Last-resort: if shapes (counts) match, align by order (useful for one-to-one datasets)
    if merged.empty:
        if len(features_df) == len(scores_df) and len(features_df) > 0:
            print("Attempting index-alignment fallback (features and scores have same length).")
            aligned = features_df.copy().reset_index(drop=True)
            aligned["human_score"] = scores_df["human_score"].reset_index(drop=True)
            merged = aligned
            print(f"Index-aligned result shape: {merged.shape}")

    if merged.empty:
        print("Sample feature utt_ids:", features_df["utt_id_raw"].head(10).tolist())
        print("Sample score utt_ids:", scores_df["utt_id_raw"].head(10).tolist())
        raise ValueError(
            "Merged dataframe is empty after multiple heuristics. "
            "Check CSV utt_ids and JSON keys for mismatches. "
            "If your feature CSV has no utt identifiers, ensure it contains 'file_path' or 'utt_id' columns."
        )

    # Select numeric features as X (exclude human_score)
    # Ensure human_score numeric
    merged["human_score"] = pd.to_numeric(merged["human_score"], errors="coerce")
    X = merged.select_dtypes(include="number").drop(columns=["human_score"], errors="ignore")
    # If no numeric columns found in X, try to coerce all other columns
    if X.shape[1] == 0:
        # try to convert all non-utt columns to numeric
        candidates = [c for c in merged.columns if c not in ("utt_id", "utt_id_raw", "utt_digits", "any_digits", "human_score", "file_path")]
        for c in candidates:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
        X = merged[candidates].select_dtypes(include="number")
    y = merged["human_score"].dropna()
    # Align X and y index
    X = X.loc[y.index]
    print(f"Final training shape X: {X.shape}, y: {y.shape}")
    return X, y

def train_models(X_train, y_train):
    """Train Linear Regression, Random Forest, and SVM"""
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "svm": SVR()
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        print(f"✅ {name} trained")
    return models

def save_models(models, models_dir: Path):
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, models_dir / f"{name}.pkl")
        print(f"Saved {name} to {models_dir / f'{name}.pkl'}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / "data/processed"
    models_dir = project_root / "models"
    score_json = project_root / "data/raw/score.json"

    # Load train data
    X_train, y_train = load_data(processed_dir / "features_train_norm.csv", score_json)

    # Train models
    models = train_models(X_train, y_train)

    # Save trained models
    save_models(models, models_dir)