import streamlit as st
import pandas as pd
import numpy as np
import librosa
import joblib

# -------------------------
# Load Model
# -------------------------
model = joblib.load("model.pkl")

# -------------------------
# Feature Extraction
# -------------------------
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)

    pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

    energy = np.mean(librosa.feature.rms(y=audio))

    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)

    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sample_rate
        )
    )

    features = [pitch, energy, zcr, spectral_centroid]
    features.extend(mfcc_mean)

    return np.array(features).reshape(1, -1)


# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="VeriVoice AI",
    page_icon="🎤",
    layout="wide"
)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.title("🎤 VeriVoice AI")

    st.divider()

    st.subheader("📋 Project Details")
    st.write("**Project:** Speech Pattern Analysis")
    st.write("**Dataset:** RAVDESS")
    st.write("**Model:** Random Forest")
    st.write("**Language:** Python")

    st.divider()

    st.subheader("⚠ Disclaimer")
    st.info(
        "This application analyzes acoustic speech features.\n\n"
        "It is NOT a lie detector."
    )

    st.divider()

    st.subheader("👨‍💻 Developer")
    st.write("Akash")

# -------------------------
# Main Page
# -------------------------
st.title("🎤 VeriVoice AI")

st.write(
    "Upload a WAV file and let the AI analyze the speech characteristics."
)

uploaded_file = st.file_uploader(
    "Choose a WAV file",
    type=["wav"]
)

# -------------------------
# Prediction
# -------------------------
if uploaded_file is not None:

    with open("uploaded.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ Audio uploaded successfully!")

    st.audio("uploaded.wav")

    features = extract_features("uploaded.wav")

    prediction = model.predict(features)

    confidence = np.max(
        model.predict_proba(features)
    ) * 100

    emotion_map = {
        "01": "Neutral",
        "02": "Calm",
        "03": "Happy",
        "04": "Sad",
        "05": "Angry",
        "06": "Fearful",
        "07": "Disgust",
        "08": "Surprised"
    }

    predicted_emotion = emotion_map.get(
        str(prediction[0]).zfill(2),
        "Unknown"
    )

    st.markdown("---")
    st.subheader("🤖 AI Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "😊 Emotion",
            predicted_emotion
        )

    with col2:
        st.metric(
            "🎯 Confidence",
            f"{confidence:.2f}%"
        )

    pitch = features[0][0]
    energy = features[0][1]
    zcr = features[0][2]
    centroid = features[0][3]

    st.markdown("---")
    st.subheader("📊 Speech Features")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("🎵 Pitch", f"{pitch:.2f}")
        st.metric("⚡ Energy", f"{energy:.4f}")

    with c2:
        st.metric("📈 Zero Crossing Rate", f"{zcr:.4f}")
        st.metric("🌊 Spectral Centroid", f"{centroid:.2f}")

    st.markdown("---")
    st.subheader("📈 Feature Visualization")

    feature_df = pd.DataFrame({
        "Feature": [
            "Pitch",
            "Energy",
            "Zero Crossing Rate",
            "Spectral Centroid"
        ],
        "Value": [
            pitch,
            energy,
            zcr,
            centroid
        ]
    })

    st.bar_chart(feature_df.set_index("Feature"))
