import streamlit as st
import numpy as np
import pandas as pd
import librosa
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# =============== CONFIG ===============
st.set_page_config(page_title="Pronunciation Evaluation", page_icon="🎙️", layout="centered")
SAMPLE_RATE = 22050
N_MFCC = 13

# =============== UTILS ===============
def extract_features(audio_path):
    """Extract MFCC, pitch, energy, spectral features, tempo"""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # Pitch
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C1'),
                                fmax=librosa.note_to_hz('C8'))
        pitch_mean = np.nanmean(f0)
        pitch_std = np.nanstd(f0)

        # Energy
        rms = librosa.feature.rms(y=y)
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)

        # Spectral
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        centroid_mean, centroid_std = np.mean(centroid), np.std(centroid)
        bandwidth_mean, bandwidth_std = np.mean(bandwidth), np.std(bandwidth)

        # Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        # Create DataFrame row
        features = np.hstack([
            mfcc_mean, mfcc_std,
            pitch_mean, pitch_std,
            energy_mean, energy_std,
            centroid_mean, centroid_std,
            bandwidth_mean, bandwidth_std,
            tempo, 0  # tempo_std placeholder
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
        return pd.DataFrame([features], columns=columns)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None


def safe_load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"⚠️ Could not load {path.name}: {e}")
        return None


def align_features_to_model(X, model):
    """Ensure input features match training features."""
    if X is None or X.empty:
        return pd.DataFrame(np.zeros((1, 1)), columns=["placeholder"])  # fallback

    if not hasattr(model, "feature_names_in_"):
        return X  # for older sklearn versions

    model_feats = list(model.feature_names_in_)
    X_aligned = pd.DataFrame(index=X.index)

    # Copy or fill missing features
    for f in model_feats:
        X_aligned[f] = X[f] if f in X.columns else 0.0

    # Handle invalids
    X_aligned = X_aligned.replace([np.nan, np.inf, -np.inf], 0)

    # Ensure at least one sample row
    if X_aligned.shape[0] == 0:
        X_aligned.loc[0] = [0.0] * len(model_feats)

    return X_aligned



# =============== STREAMLIT APP ===============
st.title("🎙️ Pronunciation Evaluation System")
st.write("Upload or record your pronunciation audio, and get model-based evaluation scores.")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    # Save temporary file
    temp_path = Path("temp.wav")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Speech recognition (optional)
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.AudioFile(str(temp_path)) as source:
        audio_data = r.record(source)
    try:
        text = r.recognize_google(audio_data)
        st.write("🗒️ Recognized Speech:")
        st.success(text)
    except Exception as e:
        st.warning(f"Speech recognition failed: {e}")
        text = ""

    # Feature extraction
    X_new = extract_features(temp_path)
    if X_new is not None:
        st.success("✅ Features extracted successfully!")

        models_dir = Path(__file__).resolve().parent.parent / "models"
        models = {
            "linear_regression": safe_load_model(models_dir / "linear_regression.pkl"),
            "random_forest": safe_load_model(models_dir / "random_forest.pkl"),
            "svm": safe_load_model(models_dir / "svm.pkl")
        }

        results = []
        for name, model in models.items():
            if model is None:
                continue

            try:
                X_aligned = align_features_to_model(X_new, model)
                pred = model.predict(X_aligned)[0]
                results.append((name, pred))
            except Exception as e:
                st.warning(f"⚠️ Error predicting with {name}: {e}")

        if results:
            pred_df = pd.DataFrame(results, columns=["Model", "Predicted Score"])
            st.write("🤖 Predictions from Trained Models")
            st.dataframe(pred_df, hide_index=True)

            # Plot
            fig, ax = plt.subplots()
            ax.bar(pred_df["Model"], pred_df["Predicted Score"], color="lightcoral")
            ax.set_ylabel("Predicted Score")
            ax.set_title("Model Predictions")
            st.pyplot(fig)
        else:
            st.error("No predictions generated — all models failed.")
    else:
        st.error("Feature extraction failed.")
