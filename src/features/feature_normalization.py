# src/features/feature_normalization.py

import pandas as pd
from pathlib import Path

def zscore_normalize(train_csv: Path, test_csv: Path, output_dir: Path):
    """Normalize numeric features using Z-score normalization (train stats)."""
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Explicitly define feature columns (exclude identifiers only)
    exclude_cols = ["file_path", "utt_id"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # Compute mean and std on training data
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1)  # avoid divide-by-zero

    # Apply normalization
    train_norm = train_df.copy()
    test_norm = test_df.copy()

    train_norm[feature_cols] = (train_df[feature_cols] - mean) / std
    test_norm[feature_cols] = (test_df[feature_cols] - mean) / std

    # Ensure identical columns and order
    test_norm = test_norm[train_norm.columns]

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    train_norm.to_csv(output_dir / "features_train_norm.csv", index=False)
    test_norm.to_csv(output_dir / "features_test_norm.csv", index=False)
    print(f"✅ Normalized features saved to {output_dir}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / "data/processed"

    zscore_normalize(
        processed_dir / "features_train.csv",
        processed_dir / "features_test.csv",
        processed_dir
    )
