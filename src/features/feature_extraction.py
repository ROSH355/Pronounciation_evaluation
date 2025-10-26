# src/features/feature_extraction.py

import librosa
import numpy as np
import pandas as pd
from pathlib import Path

# Constants
SAMPLE_RATE = 22050
N_MFCC = 13

def extract_features(audio_path):
    """
    Extracts acoustic features from a single audio file.
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

        # MFCCs (mean + std)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # Pitch (fundamental frequency using pYIN)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8')
        )
        pitch_mean = np.nanmean(f0)
        pitch_std = np.nanstd(f0)

        # Energy (RMS)
        rms = librosa.feature.rms(y=y)
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)

        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(spectral_centroid)
        centroid_std = np.std(spectral_centroid)

        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_mean = np.mean(spectral_bandwidth)
        bandwidth_std = np.std(spectral_bandwidth)

        # Tempo (approx speaking rate)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        tempo_mean = tempo
        tempo_std = 0

        # Combine features
        features = np.hstack([
            mfccs_mean, mfccs_std,
            pitch_mean, pitch_std,
            energy_mean, energy_std,
            centroid_mean, centroid_std,
            bandwidth_mean, bandwidth_std,
            tempo_mean, tempo_std
        ])
        return features

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def extract_and_save_features(dataset_wave_dir, output_csv):
    """
    Extracts features from all .wav files in a dataset folder and saves to CSV.
    """
    data = []
    for audio_file in Path(dataset_wave_dir).rglob("*.wav"):
        print(f"Processing {audio_file}...")
        features = extract_features(audio_file)
        if features is not None:
            data.append([str(audio_file)] + list(features))

    # Define column names
    columns = [
        "file_path",
        *[f"mfcc_{i+1}_mean" for i in range(N_MFCC)],
        *[f"mfcc_{i+1}_std" for i in range(N_MFCC)],
        "pitch_mean", "pitch_std",
        "energy_mean", "energy_std",
        "centroid_mean", "centroid_std",
        "bandwidth_mean", "bandwidth_std",
        "tempo_mean", "tempo_std"
    ]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"✅ Features saved to {output_csv}")

if __name__ == "__main__":
    # Automatically set project paths
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_train_wave = project_root / "data/raw/train/WAVE"
    raw_test_wave = project_root / "data/raw/test/WAVE"
    processed_dir = project_root / "data/processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Output CSV paths
    train_csv = processed_dir / "features_train.csv"
    test_csv = processed_dir / "features_test.csv"

    # Extract features
    extract_and_save_features(raw_train_wave, train_csv)
    extract_and_save_features(raw_test_wave, test_csv)
