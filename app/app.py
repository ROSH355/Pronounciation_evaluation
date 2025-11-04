import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def extract_features(audio_path):
    """Extract MFCC, pitch, energy, spectral, and tempo features."""
    y, sr = librosa.load(audio_path, sr=None)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)

    # Pitch
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    pitches = pitch[pitch > 0]
    pitch_mean = np.mean(pitches) if len(pitches) > 0 else 0
    pitch_std = np.std(pitches) if len(pitches) > 0 else 0

    # Energy
    energy = librosa.feature.rms(y=y)[0]
    energy_mean = np.mean(energy)
    energy_std = np.std(energy)

    # Spectral
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    bandwidth_mean = np.mean(bandwidth)
    bandwidth_std = np.std(bandwidth)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_mean = np.mean(tempo)
    tempo_std = np.std(tempo)

    # Combine features
    features = np.concatenate([
        mfcc_means, mfcc_stds,
        [pitch_mean, pitch_std,
         energy_mean, energy_std,
         centroid_mean, centroid_std,
         bandwidth_mean, bandwidth_std,
         tempo_mean, tempo_std]
    ])

    feature_names = (
        [f"mfcc_{i+1}_mean" for i in range(13)] +
        [f"mfcc_{i+1}_std" for i in range(13)] +
        ["pitch_mean", "pitch_std", "energy_mean", "energy_std",
         "centroid_mean", "centroid_std",
         "bandwidth_mean", "bandwidth_std",
         "tempo_mean", "tempo_std"]
    )

    return pd.DataFrame([features], columns=feature_names)


def align_features_to_model(X, model_feature_names=None):
    """Ensure input features match the trained model features."""
    if X is None or X.empty:
        return pd.DataFrame(np.zeros((1, 1)), columns=["placeholder"])

    if not model_feature_names:
        return X

    X_aligned = pd.DataFrame(index=X.index)
    for f in model_feature_names:
        X_aligned[f] = X[f] if f in X.columns else 0.0

    X_aligned = X_aligned.replace([np.nan, np.inf, -np.inf], 0.0)
    if X_aligned.shape[0] == 0:
        X_aligned.loc[0] = [0.0] * len(model_feature_names)
    return X_aligned


def load_models():
    """Load models from models/ directory."""
    models = {}
    for file in MODELS_DIR.glob("*.pkl"):
        try:
            loaded = joblib.load(file)
            if isinstance(loaded, dict) and "model" in loaded:
                models[file.stem] = {
                    "model": loaded["model"],
                    "feature_names": loaded.get("feature_names")
                }
            else:
                models[file.stem] = {
                    "model": loaded,
                    "feature_names": getattr(loaded, "feature_names_in_", None)
                }
        except Exception as e:
            st.error(f"⚠ Failed to load {file.name}: {e}")
    return models


def transcribe_audio(audio_path):
    """Transcribe audio using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(str(audio_path)) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "⚠ Could not understand the audio clearly."
    except sr.RequestError:
        return "⚠ Speech recognition service unavailable."


def plot_metrics(metrics_dict):
    """Visualize model evaluation metrics."""
    df = pd.DataFrame(metrics_dict).T
    st.subheader("📊 Model Evaluation Metrics")
    st.bar_chart(df)

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Pronunciation Evaluation", layout="wide")
st.title("🗣 Pronunciation Evaluation Dashboard")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["🎧 Predict Score", "📈 View Model Metrics"])

# ---------------------------------------------------------
# Page 1: Pronunciation Prediction
# ---------------------------------------------------------
if page == "🎧 Predict Score":
    st.subheader("🎙 Evaluate Your Pronunciation")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎧 Upload a .wav file")
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

    with col2:
        st.markdown("#### 🎤 Record your voice")
        audio_bytes = mic_recorder(start_prompt="🎙 Start Recording", stop_prompt="⏹ Stop Recording", just_once=True)

    audio_data = None
    if uploaded_file is not None:
        audio_data = uploaded_file.read()
        st.info("Using uploaded file.")
    elif audio_bytes is not None:
        audio_data = audio_bytes
        st.info("Using recorded audio.")

    if audio_data:
        tmp_path = Path("temp_audio.wav")
        with open(tmp_path, "wb") as f:
            f.write(audio_data)
        st.audio(str(tmp_path))

        with st.spinner("Transcribing audio..."):
            transcript = transcribe_audio(tmp_path)
            st.markdown("🗒 **Recognized Speech:**")
            st.write(transcript)

        with st.spinner("Extracting features..."):
            X = extract_features(tmp_path)
            st.success("✅ Features extracted successfully!")

        models = load_models()
        if not models:
            st.error("⚠ No trained models found in the 'models/' directory.")
        else:
            preds = {}
            st.subheader("🤖 Model Predictions")
            for name, entry in models.items():
                try:
                    model = entry["model"]
                    feat_names = entry["feature_names"]
                    X_aligned = align_features_to_model(X, feat_names)
                    preds[name] = float(model.predict(X_aligned)[0])
                except Exception as e:
                    preds[name] = None
                    st.error(f"Error predicting with {name}: {e}")

            pred_df = pd.DataFrame(
                [(m, s) for m, s in preds.items() if s is not None],
                columns=["Model", "Predicted Score"]
            )

            if not pred_df.empty:
                st.dataframe(pred_df)
                fig, ax = plt.subplots()
                ax.bar(pred_df["Model"], pred_df["Predicted Score"], color="skyblue")
                ax.set_ylabel("Predicted Pronunciation Score")
                ax.set_ylim(0, 10)
                st.pyplot(fig)
            else:
                st.warning("⚠ No valid predictions could be generated.")

        # Clean up temp audio
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ---------------------------------------------------------
# Page 2: Model Metrics
# ---------------------------------------------------------
elif page == "📈 View Model Metrics":
    st.subheader("📉 Model Evaluation Results")
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        plot_metrics(metrics)
    else:
        st.error("No metrics.json found. Please run evaluate.py first.")
