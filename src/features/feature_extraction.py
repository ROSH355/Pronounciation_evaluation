# src/features/feature_extraction.py
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

SAMPLE_RATE = 22050
N_MFCC = 13
MAX_FILES = 200  # ⏱️ limit number of files for quick testing

def extract_features(audio_path):
    """Extract MFCC, pitch, energy, spectral features, tempo and return as DataFrame"""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # Pitch (pYIN)
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C1'),
            fmax=librosa.note_to_hz('C8')
        )
        pitch_mean = np.nanmean(f0)
        pitch_std = np.nanstd(f0)

        # Energy (RMS)
        rms = librosa.feature.rms(y=y)
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)

        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        centroid_mean, centroid_std = np.mean(centroid), np.std(centroid)
        bandwidth_mean, bandwidth_std = np.mean(bandwidth), np.std(bandwidth)

        # Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        # Combine features
        features = np.hstack([
            mfcc_mean, mfcc_std,
            pitch_mean, pitch_std,
            energy_mean, energy_std,
            centroid_mean, centroid_std,
            bandwidth_mean, bandwidth_std,
            tempo, 0  # placeholder for tempo_std
        ])

        columns = (
            [f"mfcc_{i+1}_mean" for i in range(N_MFCC)] +
            [f"mfcc_{i+1}_std" for i in range(N_MFCC)] +
            ["pitch_mean", "pitch_std",
             "energy_mean", "energy_std",
             "centroid_mean", "centroid_std",
             "bandwidth_mean", "bandwidth_std",
             "tempo_mean", "tempo_std"]
        )

        df = pd.DataFrame([features], columns=columns)
        return df

    except Exception as e:
        print(f"⚠️ Error processing {audio_path}: {e}")
        return None


def extract_features_from_folder(wave_dir: Path, output_csv: Path):
    """Extract features from all WAV files in folder"""
    wav_files = list(wave_dir.rglob("*.wav")) + list(wave_dir.rglob("*.WAV"))
    print(f"🔍 Found {len(wav_files)} .wav files under {wave_dir}")

    # Limit files for faster testing
    if len(wav_files) > MAX_FILES:
        print(f"⚡ Processing only the first {MAX_FILES} files for quicker testing...")
        wav_files = wav_files[:MAX_FILES]

    data = []
    for idx, audio_file in enumerate(wav_files, start=1):
        df = extract_features(audio_file)
        if df is not None and not df.empty:
            row = [str(audio_file)] + list(df.iloc[0])
            data.append(row)

        # Progress update every 25 files
        if idx % 25 == 0:
            print(f"⏳ Processed {idx}/{len(wav_files)} files...")

    columns = (
        ["file_path"] +
        [f"mfcc_{i+1}_mean" for i in range(N_MFCC)] +
        [f"mfcc_{i+1}_std" for i in range(N_MFCC)] +
        ["pitch_mean", "pitch_std",
         "energy_mean", "energy_std",
         "centroid_mean", "centroid_std",
         "bandwidth_mean", "bandwidth_std",
         "tempo_mean", "tempo_std"]
    )

    df_all = pd.DataFrame(data, columns=columns)
    df_all["utt_id"] = df_all["file_path"].apply(lambda x: Path(x).stem)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_csv, index=False)
    print(f"✅ Saved features CSV to {output_csv}")
    print(f"✅ Extracted {len(df_all)} files.")
    return df_all


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / "data/processed"
    wave_dir = project_root / "data/raw/WAVE"

    extract_features_from_folder(wave_dir, processed_dir / "features_train.csv")
