import json
import pandas as pd
from pathlib import Path

def create_metadata(json_path, output_csv):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    for utt_id, info in data.items():
        records.append({
            "id": utt_id,
            "audio_path": str(Path(json_path).parent / info["path"]),
            "text": info.get("text", ""),
            "score": info.get("score", None)
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"✅ Metadata saved to {output_csv} with {len(df)} samples")

# Example usage
if __name__ == "__main__":
    create_metadata("data/raw/train/all_info.json", "data/raw/metadata_train.csv")
    create_metadata("data/raw/test/all_info.json", "data/raw/metadata_test.csv")
