"""
STEP 7: Streamlit Deployment Dashboard
-----------------------------------------
Deploy this to Streamlit Cloud (streamlit.io) after training.

BEFORE DEPLOYING, make sure these 3 files (produced by
2_train_and_evaluate.py) are in the same folder as this app.py:
    - fall_detection_model.h5
    - label_encoder.pkl
    - scaler.pkl

requirements.txt for Streamlit Cloud should include:
    streamlit
    tensorflow
    mediapipe
    opencv-python-headless
    scikit-learn
    pandas
    numpy
    matplotlib
    pillow
"""

import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Elderly Fall Detection System", layout="wide")

# ---------------- Load model + preprocessing ----------------
@st.cache_resource
def load_artifacts():
    model = load_model("fall_detection_model.h5")
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, le, scaler

model, le, scaler = load_artifacts()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# ---------------- Session state for running counts ----------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of (label, confidence)

# ---------------- Helper: run full pipeline on one frame ----------------
def predict_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if not results.pose_landmarks:
        return None, None, frame_bgr

    row = []
    for lm in results.pose_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    features = scaler.transform([row])
    features_c = features.reshape(1, features.shape[1], 1)

    probs = model.predict(features_c, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    label = le.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    annotated = frame_bgr.copy()
    mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return label, confidence, annotated


# ---------------- UI ----------------
st.title("🏥 AI-Powered Elderly Fall Detection Dashboard")
st.caption("Upload an image or video to run pose estimation and fall/activity classification.")

col_upload, col_stats = st.columns([2, 1])

with col_upload:
    file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov"])

    if file is not None:
        if file.type.startswith("image"):
            image = Image.open(file).convert("RGB")
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            label, confidence, annotated = predict_frame(frame)

            if label is None:
                st.warning("No person detected in this image.")
            else:
                st.session_state.history.append((label, confidence))
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                          caption="Pose Estimation Output", use_container_width=True)

                if label.lower() == "fall detected" or label.lower() == "fall":
                    st.error(f"🚨 EMERGENCY ALERT: Fall Detected! (confidence {confidence:.1%})")
                else:
                    st.success(f"Activity: **{label}** (confidence {confidence:.1%})")

        elif file.type.startswith("video"):
            tfile_path = "temp_upload.mp4"
            with open(tfile_path, "wb") as f:
                f.write(file.read())

            cap = cv2.VideoCapture(tfile_path)
            frame_placeholder = st.empty()
            alert_placeholder = st.empty()
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % 5 != 0:   # sample every 5th frame for speed
                    continue

                label, confidence, annotated = predict_frame(frame)
                if label is not None:
                    st.session_state.history.append((label, confidence))
                    frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                              caption=f"Frame {frame_count} — {label} ({confidence:.1%})",
                                              use_container_width=True)
                    if label.lower() in ("fall detected", "fall"):
                        alert_placeholder.error(f"🚨 EMERGENCY ALERT: Fall Detected at frame {frame_count}!")

            cap.release()

with col_stats:
    st.subheader("📊 Monitoring Analytics")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history, columns=["label", "confidence"])
        total = len(hist_df)
        falls = (hist_df["label"].str.lower().isin(["fall detected", "fall"])).sum()
        normal = total - falls

        st.metric("Total Activities Detected", total)
        st.metric("Fall Detected Count", falls)
        st.metric("Normal Activity Count", normal)
        st.metric("Avg. Prediction Confidence", f"{hist_df['confidence'].mean():.1%}")

        st.write("Activity distribution:")
        counts = hist_df["label"].value_counts()
        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.info("Upload an image or video to see analytics here.")

    if st.button("Reset session stats"):
        st.session_state.history = []
        st.experimental_rerun()
