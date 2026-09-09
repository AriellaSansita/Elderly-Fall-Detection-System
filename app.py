"""
Fall / Activity Detection — Streamlit App
==========================================
Loads the CNN trained in the companion Colab notebook
(fall_detection_model.h5 + class_names.txt) and lets a user upload or
capture an image to classify it into one of the 5 activity classes:
Fall, Walking, Sitting, Standing, Normal.

Optionally overlays the MediaPipe BlazePose skeleton on the image.

Run with:
    streamlit run app.py

Expected files in the same folder as this script:
    fall_detection_model.h5
    class_names.txt
    pose_landmarker.task   (auto-downloaded on first run if missing)
"""

import os
import urllib.request

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
IMG_SIZE = (128, 128)
MODEL_PATH = "fall_detection_model.h5"
CLASS_NAMES_PATH = "class_names.txt"
POSE_MODEL_PATH = "pose_landmarker.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# Standard 33-point BlazePose skeleton connections (stable across versions)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

st.set_page_config(page_title="Activity / Fall Detection", page_icon="🏃", layout="centered")


# ----------------------------------------------------------------------
# Cached resource loaders
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading classification model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file '{MODEL_PATH}' not found. Place it next to app.py "
            "(it's produced by the training notebook)."
        )
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        st.error(f"'{CLASS_NAMES_PATH}' not found. Place it next to app.py.")
        st.stop()
    with open(CLASS_NAMES_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]


@st.cache_resource(show_spinner="Loading pose model...")
def load_pose_detector():
    if not os.path.exists(POSE_MODEL_PATH):
        try:
            urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        except Exception as e:
            st.warning(f"Could not download pose model automatically: {e}")
            return None
    base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    return vision.PoseLandmarker.create_from_options(options)


# ----------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------
def classify_image(model, class_names, pil_image: Image.Image):
    """Resize/normalize a PIL image and run the CNN. Returns (label, probs dict)."""
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dim

    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    label = class_names[pred_idx]
    prob_dict = {name: float(p) for name, p in zip(class_names, probs)}
    return label, prob_dict


def draw_pose_on_array(detector, bgr_image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Runs MediaPipe pose detection on a BGR numpy image and draws the skeleton.
    Returns (annotated_bgr_image, person_detected)."""
    if detector is None:
        return bgr_image, False

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    annotated = bgr_image.copy()
    if not result.pose_landmarks:
        return annotated, False

    h, w, _ = annotated.shape
    landmarks = result.pose_landmarks[0]  # first detected person

    points = []
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
        cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(annotated, points[start_idx], points[end_idx], (255, 0, 0), 2)

    return annotated, True


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🏃 Activity / Fall Detection")
st.caption("Upload a photo or use your camera to classify: Fall, Walking, Sitting, Standing, Normal.")

with st.sidebar:
    st.header("Options")
    show_pose = st.checkbox("Overlay pose skeleton", value=True)
    show_probs = st.checkbox("Show class probabilities", value=True)
    st.markdown("---")
    st.caption(
        "Model expects: fall_detection_model.h5 and class_names.txt "
        "in the app folder (from the training notebook)."
    )

model = load_model()
class_names = load_class_names()
pose_detector = load_pose_detector() if show_pose else None

tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Camera"])

image_source = None
with tab_upload:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file)

with tab_camera:
    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        image_source = Image.open(camera_file)

if image_source is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        st.image(image_source, use_container_width=True)

    with st.spinner("Classifying..."):
        label, probs = classify_image(model, class_names, image_source)

    bgr = cv2.cvtColor(np.array(image_source.convert("RGB")), cv2.COLOR_RGB2BGR)

    if show_pose:
        with st.spinner("Detecting pose..."):
            annotated_bgr, person_found = draw_pose_on_array(pose_detector, bgr)
        display_img = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    else:
        display_img = np.array(image_source.convert("RGB"))
        person_found = None

    with col2:
        st.subheader("Result")
        st.image(display_img, use_container_width=True)
        if show_pose and person_found is False:
            st.caption("⚠️ No person detected for pose overlay.")

    st.markdown("---")
    confidence = probs[label]
    if label.lower() == "fall":
        st.error(f"### Prediction: **{label}**  ({confidence:.1%} confidence)")
    else:
        st.success(f"### Prediction: **{label}**  ({confidence:.1%} confidence)")

    if show_probs:
        st.bar_chart(probs)
else:
    st.info("Upload an image or take a photo to get a prediction.")
