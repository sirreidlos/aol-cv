from typing import List, Tuple
import streamlit as st
import numpy as np
import cv2

from acf.inference import (
    detect_multiscale_fast,
    get_scales_octave_based,
    non_max_suppression,
    visualize_detections,
    visualize_feature_map,
)
from acf.model import ACFDetector
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


acf_model_map = {
    "ACF+MLP": MODELS_DIR / "muct_mlp.pkl",
    "ACF+LightGBM": MODELS_DIR / "muct_gbm_sc.pkl",
    "ACF+AdaBoost": MODELS_DIR / "muct_ada.pkl",
}


@st.cache_resource
def load_acf_detector(model_path: Path) -> ACFDetector:
    detector = ACFDetector()
    detector.load(model_path)
    return detector


def main():
    run_the_app()


def run_the_app():
    st.sidebar.header("Model")

    model_choice = st.sidebar.selectbox(
        "Select detector",
        [
            "ACF+MLP",
            "ACF+LightGBM",
            "ACF+AdaBoost",
            "Viola-Jones",
        ],
    )

    detector = None
    use_vj = False

    if model_choice.startswith("ACF"):
        detector = load_acf_detector(acf_model_map[model_choice])
    else:
        use_vj = True

    st.sidebar.header("Input")

    uploaded_file = st.sidebar.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "webp"],
    )

    st.sidebar.header("Detection thresholds")
    score_threshold = st.sidebar.slider("Score threshold", 0.0, 1.0, 0.5, 0.01)
    nms_threshold = st.sidebar.slider("NMS threshold", 0.0, 1.0, 0.3, 0.01)

    st.sidebar.header("Inference parameters")
    stride = st.sidebar.number_input(
        "Stride", min_value=1, max_value=64, value=8, step=1
    )
    batch_size = st.sidebar.number_input(
        "Batch size", min_value=1, max_value=16384, value=32, step=1
    )

    st.sidebar.header("Scale parameters")
    num_scales = st.sidebar.number_input(
        "Scales per octave", min_value=1, max_value=16, value=8, step=1
    )
    num_octaves = st.sidebar.number_input(
        "Number of octaves", min_value=1, max_value=8, value=2, step=1
    )

    st.sidebar.header("Window size")
    min_window_width = st.sidebar.number_input(
        "Minimum window width", min_value=8, max_value=512, value=24, step=1
    )
    min_window_height = st.sidebar.number_input(
        "Minimum window height", min_value=8, max_value=512, value=24, step=1
    )

    if uploaded_file is None:
        return

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption="Uploaded image", width="stretch")

    cache_key = (
        model_choice,
        hash(uploaded_file.getvalue()),
        stride,
        batch_size,
        num_scales,
        num_octaves,
        min_window_width,
        min_window_height,
    )

    if (
        "all_detections" not in st.session_state
        or st.session_state.get("cache_key") != cache_key
    ):
        st.session_state.cache_key = cache_key

        if use_vj:
            faces = run_inference_vj_from_image(image)
            # convert to (x, y, w, h, score)
            st.session_state.all_detections = [
                (x, y, w, h, 1.0) for (x, y, w, h) in faces
            ]
        else:
            st.session_state.all_detections = run_inference_mlp(
                detector=detector,
                image=image,
                stride=stride,
                batch_size=batch_size,
                num_scales=num_scales,
                num_octaves=num_octaves,
                min_ds=(min_window_width, min_window_height),
            )

    assert st.session_state.all_detections is not None
    all_boxes = np.array([det[:4] for det in st.session_state.all_detections])
    all_scores = np.array([det[4] for det in st.session_state.all_detections])

    score_filtered_indices = [
        i for i, s in enumerate(all_scores) if s >= score_threshold
    ]
    boxes = all_boxes[score_filtered_indices]
    scores = all_scores[score_filtered_indices]

    if len(boxes) > 0:
        keep_indices = non_max_suppression(boxes, scores, nms_threshold)
        filtered_detections = [(*boxes[i], scores[i]) for i in keep_indices]
    else:
        filtered_detections = []

    vis_image = visualize_detections(image.copy(), filtered_detections)
    vis_feat = None

    if filtered_detections and not use_vj:
        assert detector
        first_det = filtered_detections[0]
        x, y, w, h, score = first_det
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        features = detector.extract_features(image, (x, y, w, h))
        feature_map = features.reshape(
            (detector.feature_resolution, detector.feature_resolution, 10)
        )

        detected_window = image[y : y + h, x : x + w]
        vis_feat = visualize_feature_map(feature_map, 1, score, detected_window)

    st.image(vis_image, caption="Detected faces", width="stretch")

    if vis_feat is not None:
        st.image(vis_feat, caption="Example feature map", width="stretch")


def run_inference_vj_from_image(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
    )

    return faces


def run_inference_mlp(
    detector,
    image,
    stride,
    batch_size,
    num_scales,
    num_octaves,
    min_ds,
) -> List[Tuple[int, int, int, int, float]]:
    progress = st.progress(0.0)
    status = st.empty()

    def progress_cb(done, total):
        frac = done / total
        progress.progress(frac)
        status.text(f"Scanning windows: {done}/{total}")

    scales = get_scales_octave_based(num_scales, num_octaves, None)
    max_ds = (int(min_ds[0] * scales[-1]), int(min_ds[1] * scales[-1]))

    detections = detect_multiscale_fast(
        detector=detector,
        image=image,
        window_size=max_ds,
        scales=scales,
        stride=stride,
        score_threshold=0.0,
        nms_threshold=1.0,
        batch_size=batch_size,
        progress_cb=progress_cb,
    )

    progress.progress(1.0)
    status.text("Detection complete")

    return detections


if __name__ == "__main__":
    main()
