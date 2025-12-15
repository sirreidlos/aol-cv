import streamlit as st
from streamlit_image_comparison import image_comparison
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


def main():
    @st.cache_resource
    def load_detector() -> ACFDetector:
        detector = ACFDetector()
        detector.load("./models/cleanset.pkl")
        return detector

    detector = load_detector()
    run_the_app(detector)


def run_the_app(detector: ACFDetector):
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
        "Batch size", min_value=1, max_value=512, value=32, step=1
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

    if "all_detections" not in st.session_state or st.session_state.get(
        "image_hash"
    ) != hash(uploaded_file.getvalue()):
        st.session_state.image_hash = hash(uploaded_file.getvalue())
        st.session_state.all_detections = run_inference(
            detector=detector,
            image=image,
            stride=stride,
            batch_size=batch_size,
            num_scales=num_scales,
            num_octaves=num_octaves,
            min_ds=(min_window_width, min_window_height),
        )

    assert st.session_state.all_detections is not None
    # filtered_detections = [
    #     det for det in st.session_state.all_detections if det[4] >= score_threshold
    # ]

    all_boxes = np.array([det[:4] for det in st.session_state.all_detections])
    all_scores = np.array([det[4] for det in st.session_state.all_detections])

    # Filter by score first
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

    if filtered_detections:
        first_det = filtered_detections[0]
        x, y, w, h, score = first_det
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        features = detector.extract_features(image, (x, y, w, h))
        feature_map = features.reshape(
            (detector.feature_resolution, detector.feature_resolution, 10)
        )
        vis_feat = visualize_feature_map(feature_map, 1, score)

    st.image(vis_image, caption="Detected faces", width="stretch")

    if vis_feat is not None:
        st.image(vis_feat, caption="Example feature map", width="stretch")


def run_inference(
    detector,
    image,
    stride,
    batch_size,
    num_scales,
    num_octaves,
    min_ds,
):
    progress = st.progress(0.0)
    status = st.empty()

    def progress_cb(done, total):
        frac = done / total
        progress.progress(frac)
        status.text(f"Scanning windows: {done}/{total}")

    scales = get_scales_octave_based(num_scales, num_octaves, min_ds)

    detections = detect_multiscale_fast(
        detector=detector,
        image=image,
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
