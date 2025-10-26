# ...existing code...
import argparse
import json
from pathlib import Path

def read_map(path: Path):
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

def main():
    p = argparse.ArgumentParser(description="Create all_info.json from Kaldi-style files")
    p.add_argument("data_dir", nargs="?", default="data/raw",
                   help="directory containing splits (train/dev/test) or a specific split")
    args = p.parse_args()

    root = Path(args.data_dir)
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    # If given top-level data/raw, process each subdirectory
    if (root / "wav.scp").exists() or (root / "text").exists():
        # single split provided
        build_all_info(root)
    else:
        for sub in sorted(root.iterdir()):
            if sub.is_dir() and (sub / "wav.scp").exists():
                build_all_info(sub)

if __name__ == "__main__":
    main()
# ...existing code...