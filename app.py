import streamlit as st
from ultralytics import YOLO
from tensorflow.keras.models import load_model

import numpy as np
import cv2
import os
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Vision Assistant",
    page_icon="👁️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("👁️ AI Vision Assistant")

st.markdown(
    """
    **📷 Capture / Upload → 🤖 YOLO Object Detection → 🎨 ANN Color Detection → 🔊 Voice → 📊 Summary**
    """
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")
ANN_MODEL_PATH = os.path.join(BASE_DIR, "color_ann.keras")
SCALER_PATH = os.path.join(BASE_DIR, "color_scaler.npy")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.txt")


# ============================================================
# LOAD YOLO
# ============================================================

@st.cache_resource
def load_yolo():
    if not os.path.exists(YOLO_MODEL_PATH):
        st.error("YOLO model not found.")
        st.stop()

    return YOLO(YOLO_MODEL_PATH)


# ============================================================
# LOAD ANN
# ============================================================

@st.cache_resource
def load_ann():
    if not os.path.exists(ANN_MODEL_PATH):
        st.error("ANN model not found.")
        st.stop()

    return load_model(ANN_MODEL_PATH)


# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def load_scaler():

    if not os.path.exists(SCALER_PATH):
        st.error("Scaler file not found.")
        st.stop()

    try:

        scaler_data = np.load(
            SCALER_PATH,
            allow_pickle=True
        )

        if scaler_data.ndim == 0:
            scaler_data = scaler_data.item()

        if not isinstance(scaler_data, dict):
            st.error("Invalid scaler format.")
            st.stop()

        if "mean" not in scaler_data or "scale" not in scaler_data:
            st.error("Scaler must contain mean and scale.")
            st.stop()

        scaler_mean = np.asarray(
            scaler_data["mean"],
            dtype=np.float32
        )

        scaler_scale = np.asarray(
            scaler_data["scale"],
            dtype=np.float32
        )

        if len(scaler_mean) != 108:
            st.error(
                f"Scaler has {len(scaler_mean)} features. Expected 108."
            )
            st.stop()

        if len(scaler_scale) != 108:
            st.error(
                f"Scaler has {len(scaler_scale)} features. Expected 108."
            )
            st.stop()

        scaler_scale = np.where(
            scaler_scale == 0,
            1.0,
            scaler_scale
        )

        return scaler_mean, scaler_scale

    except Exception as e:
        st.error(f"Scaler loading error: {e}")
        st.stop()


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_resource
def load_class_names():

    if not os.path.exists(CLASS_NAMES_PATH):
        st.error("class_names.txt not found.")
        st.stop()

    with open(CLASS_NAMES_PATH, "r") as file:

        class_names = [
            line.strip()
            for line in file.readlines()
            if line.strip()
        ]

    return class_names


# ============================================================
# LOAD MODELS
# ============================================================

yolo_model = load_yolo()
ann_model = load_ann()

scaler_mean, scaler_scale = load_scaler()

class_names = load_class_names()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_color_features(image):

    image = cv2.resize(
        image,
        (6, 6)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(
        np.float32
    ) / 255.0

    features = image.flatten()

    if len(features) != 108:
        raise ValueError(
            f"Expected 108 features, got {len(features)}"
        )

    return features


# ============================================================
# ANN COLOR PREDICTION
# ============================================================

def predict_color(crop):

    try:

        features = extract_color_features(crop)

        features = features.reshape(
            1,
            -1
        )

        scaled_features = (
            features - scaler_mean
        ) / scaler_scale

        prediction = ann_model.predict(
            scaled_features,
            verbose=0
        )

        class_index = int(
            np.argmax(prediction[0])
        )

        confidence = float(
            prediction[0][class_index] * 100
        )

        if class_index < len(class_names):
            color = class_names[class_index]
        else:
            color = "unknown"

        return color, confidence

    except Exception as e:

        print("COLOR ERROR:", e)

        return "unknown", 0.0


# ============================================================
# BROWSER VOICE
# ============================================================

def browser_voice(text):

    safe_text = (
        text
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", " ")
    )

    st.components.v1.html(
        f"""
        <script>
        const text = '{safe_text}';

        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();

            const speech = new SpeechSynthesisUtterance(text);

            speech.rate = 0.9;
            speech.pitch = 1.0;
            speech.volume = 1.0;

            window.speechSynthesis.speak(speech);
        }}
        </script>
        """,
        height=0
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(image):

    output_image = image.copy()

    detections = []

    results = yolo_model(
        image,
        conf=0.35,
        verbose=False
    )

    result = results[0]

    if result.boxes is None:
        return output_image, detections

    for box in result.boxes:

        try:

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))

            x2 = min(
                image.shape[1],
                int(x2)
            )

            y2 = min(
                image.shape[0],
                int(y2)
            )

            if x2 <= x1 or y2 <= y1:
                continue

            class_id = int(
                box.cls[0]
                .cpu()
                .numpy()
            )

            object_name = yolo_model.names[class_id]

            yolo_confidence = float(
                box.conf[0]
                .cpu()
                .numpy()
                * 100
            )

            crop = image[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            color, color_confidence = predict_color(crop)

            label = f"{color} {object_name}"

            confidence_text = (
                f"YOLO: {yolo_confidence:.0f}% | "
                f"ANN: {color_confidence:.0f}%"
            )

            cv2.rectangle(
                output_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                output_image,
                label,
                (
                    x1,
                    max(35, y1 - 35)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                output_image,
                confidence_text,
                (
                    x1,
                    max(65, y1 - 8)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            detections.append({
                "Object": object_name,
                "Color": color,
                "YOLO Confidence": round(
                    yolo_confidence,
                    2
                ),
                "ANN Confidence": round(
                    color_confidence,
                    2
                )
            })

        except Exception as e:

            print(
                "DETECTION ERROR:",
                e
            )

    return output_image, detections


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    result_image,
    detections,
    voice_enabled
):

    st.markdown("---")

    st.subheader("🎯 AI Detection Result")

    st.image(
        cv2.cvtColor(
            result_image,
            cv2.COLOR_BGR2RGB
        ),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("📊 Detection Summary")

    if detections:

        df = pd.DataFrame(detections)

        total_objects = len(detections)

        avg_yolo = df[
            "YOLO Confidence"
        ].mean()

        avg_ann = df[
            "ANN Confidence"
        ].mean()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Objects Detected",
                total_objects
            )

        with col2:
            st.metric(
                "Avg YOLO Confidence",
                f"{avg_yolo:.1f}%"
            )

        with col3:
            st.metric(
                "Avg ANN Confidence",
                f"{avg_ann:.1f}%"
            )

        st.markdown("### 🔎 Detailed Results")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### 📝 AI Summary")

        voice_parts = []

        for i, detection in enumerate(
            detections,
            start=1
        ):

            sentence = (
                f"{detection['Color'].title()} "
                f"{detection['Object'].title()}"
            )

            voice_parts.append(sentence)

            st.write(
                f"**{i}. {sentence}** — "
                f"YOLO: "
                f"{detection['YOLO Confidence']:.1f}% | "
                f"Color: "
                f"{detection['ANN Confidence']:.1f}%"
            )

        if voice_enabled:

            if len(voice_parts) == 1:

                voice_message = (
                    "I detected "
                    + voice_parts[0]
                )

            else:

                voice_message = (
                    "I detected "
                    + ", ".join(voice_parts)
                )

            browser_voice(voice_message)

            st.success(
                "🔊 Voice announcement sent to your browser."
            )

    else:

        st.warning(
            "⚠️ No object was detected."
        )

        st.info(
            "Try another image with the object clearly visible."
        )


# ============================================================
# INPUT MODE
# ============================================================

st.markdown("---")

st.header("🎯 Choose Input")

tab1, tab2 = st.tabs(
    [
        "📷 Capture Photo",
        "📤 Upload Image"
    ]
)


# ============================================================
# CAMERA
# ============================================================

with tab1:

    st.subheader("📷 Take a Photo")

    st.info(
        "Use your camera to capture an image."
    )

    camera_photo = st.camera_input(
        "📸 Take a picture"
    )

    if camera_photo is not None:

        image_bytes = np.asarray(
            bytearray(
                camera_photo.getvalue()
            ),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error(
                "Unable to read camera image."
            )

        else:

            st.subheader("🖼️ Captured Image")

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

            voice_enabled = st.checkbox(
                "🔊 Enable Voice Output",
                value=True,
                key="camera_voice"
            )

            if st.button(
                "🤖 Detect Objects & Colors",
                type="primary",
                key="camera_detect"
            ):

                with st.spinner(
                    "🔍 Analyzing captured image..."
                ):

                    result_image, detections = process_image(
                        image
                    )

                display_results(
                    result_image,
                    detections,
                    voice_enabled
                )


# ============================================================
# UPLOAD IMAGE
# ============================================================

with tab2:

    st.subheader("📤 Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        key="image_upload"
    )

    if uploaded_file is not None:

        image_bytes = np.asarray(
            bytearray(
                uploaded_file.read()
            ),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error(
                "Unable to read uploaded image."
            )

        else:

            st.subheader("🖼️ Uploaded Image")

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

            voice_enabled = st.checkbox(
                "🔊 Enable Voice Output",
                value=True,
                key="upload_voice"
            )

            if st.button(
                "🤖 Detect Objects & Colors",
                type="primary",
                key="upload_detect"
            ):

                with st.spinner(
                    "🔍 Analyzing uploaded image..."
                ):

                    result_image, detections = process_image(
                        image
                    )

                display_results(
                    result_image,
                    detections,
                    voice_enabled
                )


# ============================================================
# AI PIPELINE
# ============================================================

st.markdown("---")

st.header("🧠 AI Pipeline")

st.markdown(
    """
**📷 Capture Photo / 📤 Upload Image**

↓

**🤖 YOLO Object Detection**

↓

**✂️ Object Crop**

↓

**🖼️ Resize to 6 × 6**

↓

**🔄 BGR → RGB**

↓

**📊 Normalize /255**

↓

**🔢 108 Features**

↓

**⚖️ StandardScaler**

↓

**🧠 ANN Color Classification**

↓

**🎨 Color + Object**

↓

**🔊 Browser Voice Output**

↓

**📊 Detection Summary**
"""
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

st.header("📊 Model Information")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "YOLO",
        "YOLO11n"
    )

with col2:

    st.metric(
        "ANN Features",
        "108"
    )

with col3:

    st.metric(
        "Color Classes",
        len(class_names)
    )

with col4:

    st.metric(
        "Input Size",
        "6 × 6"
    )


# ============================================================
# SUPPORTED COLORS
# ============================================================

st.markdown("---")

st.header("🎨 Supported Colors")

st.write(
    " • ".join(class_names)
)