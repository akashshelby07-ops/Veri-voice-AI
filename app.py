from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
#import sounddevice as sd 
import pandas as pd 
import streamlit as st
#import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import joblib
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)

    pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
    pitch = np.mean(pitches[pitches > 0])

    energy = np.mean(librosa.feature.rms(y=audio))

    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)

    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    )

    features = [pitch, energy, zcr, spectral_centroid]
    features.extend(mfcc_mean)

    return np.array(features).reshape(1, -1)

# Load the trained model
model = joblib.load("model.pkl")
st.set_page_config(page_title="VeriVoice AI", page_icon="🎤",layout="wide")
st.markdown("""
<style>

/* Main App */
.stApp{
    background-color:#0B1120;
    color:white;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background-color:#111827;
}

/* Title */
h1{
    color:#38BDF8;
    text-align:center;
    font-size:48px;
}

/* Subtitles */
h2,h3{
    color:white;
}

/* Buttons */
div.stButton > button{
    width:100%;
    height:55px;
    border-radius:15px;
    background:linear-gradient(90deg,#2563EB,#06B6D4);
    color:white;
    border:none;
    font-size:18px;
    font-weight:bold;
}

div.stButton > button:hover{
    background:linear-gradient(90deg,#1D4ED8,#0891B2);
    transform:scale(1.02);
    transition:0.3s;
}

/* Metric Cards */
[data-testid="stMetric"]{
    background:#1E293B;
    border-radius:15px;
    padding:15px;
}

</style>
""", unsafe_allow_html=True)
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
        "This application analyzes acoustic speech features. "
        "It is not a lie detector."
    )

    st.divider()

    st.subheader("👨‍💻 Developer")
    st.write("Akash")
    st.caption("Machine Learning Semester Project")

st.markdown("""
# 🎤 VeriVoice AI

### Professional Speech Pattern Analysis Dashboard

Analyze acoustic speech features using Machine Learning.

---
""")

st.success(
    "✅ Welcome! Record your voice and let the AI analyze its speech characteristics."
)

st.warning(
    "⚠ This tool analyzes speech patterns and emotions. "
    "It is **not** a lie detector and should not be used to determine truthfulness."

)

st.markdown("---")
st.subheader("📂 Upload Audio File")

uploaded_file = st.file_uploader(
    "Choose a WAV file",
    type=["wav"]
)

if uploaded_file is not None:

    with open("uploaded.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ Audio uploaded successfully!")

    st.audio("uploaded.wav")


    # Analyze uploaded audio
    features = extract_features("uploaded.wav")

    prediction = model.predict(features)

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

    confidence = np.max(model.predict_proba(features)) * 100

    st.markdown("---")
    st.subheader("🤖 Upload Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("😊 Emotion", predicted_emotion)

    with col2:
        st.metric("🎯 Confidence", f"{confidence:.1f}%")
# Recording Section
st.markdown("## 🎙️ Record Your Voice")

col1, col2 = st.columns([2, 1])

with col1:
    record = st.button("🎤 Start Recording", use_container_width=True)

with col2:
    st.metric("Duration", "5 sec")

duration = 5
sample_rate = 44100


#audio_file = open("sample.wav", "rb")
#audio_bytes = audio_file.read()

#st.audio(audio_bytes, format="audio/wav")
    # Extract features from the recorded audio
#features = extract_features("sample.wav")

# Predict using the trained model
#prediction = model.predict(features)

# Display the result
