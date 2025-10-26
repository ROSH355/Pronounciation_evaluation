# src/data/prepare_metadata.py

import argparse
import json
from pathlib import Path
import pandas as pd

def read_map(path: Path):
    """
    Read Kaldi-style mapping files (wav.scp, text, utt2spk)
    Returns a dictionary: key -> value
    """
    if not path.exists():
        return {}
    d = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            utt = parts[0]
            rest = " ".join(parts[1:]) if len(parts) > 1 else ""
            d[utt] = rest
    return d

def build_all_info(data_dir: Path):
    """
    Build all_info.json from wav.scp, text, utt2spk
    """
    wav = read_map(data_dir / "wav.scp")
    text = read_map(data_dir / "text")
    utt2spk = read_map(data_dir / "utt2spk")

    entries = []
    for utt, wav_path in wav.items():
        entry = {"utt_id": utt, "wav_path": wav_path}
        if text.get(utt):
            entry["text"] = text[utt]
        if utt2spk.get(utt):
            entry["speaker"] = utt2spk[utt]
        entries.append(entry)

    out = data_dir / "all_info.json"
    out.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(entries)} entries to {out}")
    return out

def json_to_csv(all_info_json: Path, output_csv: Path):
    """
    Convert all_info.json to CSV
    """
    with open(all_info_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved metadata CSV to {output_csv}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Prepare metadata CSVs from Kaldi-style files")
    parser.add_argument("data_dir", nargs="?", default="data/raw",
                        help="directory containing train/test splits")
    args = parser.parse_args()

    root = Path(args.data_dir)
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for split in ["train", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            print(f"⚠ Skip missing split: {split_dir}")
            continue

        # Step 1: create all_info.json if missing
        all_info_json = split_dir / "all_info.json"
        if not all_info_json.exists():
            print(f"Creating {all_info_json}...")
            all_info_json = build_all_info(split_dir)
        else:
            print(f"Found existing {all_info_json}")

        # Step 2: convert to CSV
        csv_file = processed_dir / f"metadata_{split}.csv"
        df = json_to_csv(all_info_json, csv_file)
        all_dfs.append(df)

    # Step 3: combined metadata.csv
    if all_dfs:
        combined_csv = processed_dir / "metadata.csv"
        pd.concat(all_dfs, ignore_index=True).to_csv(combined_csv, index=False)
        print(f"✅ Saved combined metadata CSV to {combined_csv}")

if __name__ == "__main__":
    main()