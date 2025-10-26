import pandas as pd
import json
from pathlib import Path

features_csv = "data/processed/features_train_norm.csv"
score_json = "data/raw/score.json"

# Load CSV
df = pd.read_csv(features_csv)
df["utt_id"] = df["file_path"].apply(lambda x: Path(x).stem.strip())
print("Sample CSV utt_ids:", df["utt_id"].head(10).tolist())

# Load JSON
with open(score_json, "r", encoding="utf-8") as f:
    scores = json.load(f)
json_keys = [str(k).strip() for k in scores.keys()]
print("Sample JSON keys:", json_keys[:10])

# Compare
csv_set = set(df["utt_id"])
json_set = set(json_keys)

print("CSV not in JSON:", csv_set - json_set)
print("JSON not in CSV:", json_set - csv_set)
