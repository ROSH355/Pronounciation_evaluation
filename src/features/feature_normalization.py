# src/features/feature_normalization.py

import pandas as pd
from pathlib import Path

def zscore_normalize(train_csv: Path, test_csv: Path, output_dir: Path):
    """Normalize numeric features using training set statistics"""
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Select numeric columns only
    feature_cols = train_df.select_dtypes(include='number').columns.tolist()

    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std()

    train_norm = train_df.copy()
    test_norm = test_df.copy()

    train_norm[feature_cols] = (train_df[feature_cols] - mean) / std
    test_norm[feature_cols] = (test_df[feature_cols] - mean) / std  # use train stats

    output_dir.mkdir(parents=True, exist_ok=True)
    train_norm.to_csv(output_dir / "features_train_norm.csv", index=False)
    test_norm.to_csv(output_dir / "features_test_norm.csv", index=False)

    print(f"✅ Normalized features saved to {output_dir / 'features_train_norm.csv'}")
    print(f"✅ Normalized features saved to {output_dir / 'features_test_norm.csv'}")


if __name__ == "__main__":
    # Resolve project root
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / "data/processed"

    train_csv = processed_dir / "features_train.csv"
    test_csv = processed_dir / "features_test.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Required feature CSVs not found in {processed_dir}.\n"
            "Run feature_extraction.py first!"
        )

    zscore_normalize(train_csv, test_csv, processed_dir)