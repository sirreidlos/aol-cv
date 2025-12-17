from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import cv2
from tqdm import tqdm

from acf.model import ACFDetector
from .channels import compute_channels, compute_channel_pyramid
from .preprocessing import compute_iou, generate_sliding_windows


def non_max_suppression(
    boxes: List[Tuple[int, int, int, int]], scores: List[float], iou_threshold=0.3
) -> List[int]:
    """
    out: keep_indices: Indices of boxes to keep
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = boxes[:, 2] * boxes[:, 3]

    order = scores.argsort()[::-1]

    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def compute_fast_feature_pyramid(
    channels: np.ndarray, scales: List[float], window_size: Tuple[int, int]
) -> List[Tuple[np.ndarray, float]]:
    pyramid = []
    h, w = channels.shape[:2]
    win_w, win_h = window_size

    for scale in scales:
        scaled_h, scaled_w = int(h * scale), int(w * scale)

        if scaled_h < win_h or scaled_w < win_w:
            continue

        scaled_channels = np.zeros((scaled_h, scaled_w, 10), dtype=channels.dtype)
        for c in range(10):
            scaled_channels[..., c] = cv2.resize(
                channels[..., c], (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR
            )

        pyramid.append((scaled_channels, scale))

    return pyramid


def detect_multiscale_fast(
    detector: ACFDetector,
    image: np.ndarray,
    scales=None,
    stride=8,
    score_threshold=0.5,
    nms_threshold=0.3,
    batch_size=32,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[Tuple[int, int, int, int, float]]:
    def normalize_features(detector, X: np.ndarray) -> np.ndarray:
        X = X.reshape(-1, detector.feature_resolution, detector.feature_resolution, 10)
        X = (X - detector.mean) / detector.std
        return X.reshape(
            -1, detector.feature_resolution * detector.feature_resolution * 10
        )

    if not detector.trained:
        raise ValueError("Detector must be trained before inference")

    if scales is None:
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]

    channels = compute_channels(image)
    pyramid = compute_fast_feature_pyramid(channels, scales, detector.window_size)

    detector.classifier.eval()

    all_boxes = []
    all_scores = []

    windows_per_scale = []
    total_windows = 0

    for scaled_channels, scale in pyramid:
        h, w = scaled_channels.shape[:2]
        win_w, win_h = detector.window_size

        if h < win_h or w < win_w:
            windows_per_scale.append((scaled_channels, scale, []))
            continue

        windows = generate_sliding_windows((h, w), detector.window_size, stride)
        windows_per_scale.append((scaled_channels, scale, windows))
        total_windows += len(windows)

    with tqdm(
        total=total_windows,
        desc="Detecting faces",
        unit="window",
        ncols=100,
    ) as pbar:
        for scaled_channels, scale, windows in windows_per_scale:
            if not windows:
                continue

            for i in range(0, len(windows), batch_size):
                batch_windows = windows[i : i + batch_size]
                batch_features = []

                for window in batch_windows:
                    features = detector.extract_features_from_channels(
                        scaled_channels, window
                    )
                    batch_features.append(features)

                if batch_features:
                    features_array = np.array(batch_features, dtype=np.float32)
                    features_normalized = normalize_features(detector, features_array)
                    scores = detector.classifier.infer_batch(features_normalized)

                    for idx, window in enumerate(batch_windows):
                        score = scores[idx]
                        if score > score_threshold:
                            x, y, win_w, win_h = window
                            all_boxes.append(
                                [
                                    int(x / scale),
                                    int(y / scale),
                                    int(win_w / scale),
                                    int(win_h / scale),
                                ]
                            )
                            all_scores.append(float(score))

                if progress_cb is not None:
                    # callback for progress if needed
                    progress_cb(pbar.n, pbar.total)
                pbar.update(len(batch_windows))

    if all_boxes:
        keep_indices = non_max_suppression(all_boxes, all_scores, nms_threshold)
        return [(*all_boxes[i], all_scores[i]) for i in keep_indices]

    return []


def get_scales_octave_based(n_per_oct=8, n_oct_up=0, min_ds=(16, 16), max_scale=None):
    scales = []
    scale_factor = 2 ** (-1.0 / n_per_oct)

    current_scale = 2**n_oct_up

    while True:
        scales.append(current_scale)
        if max_scale and current_scale < max_scale:
            break
        current_scale *= scale_factor

        if (
            current_scale * min_ds[0] < min_ds[0]
            or current_scale * min_ds[1] < min_ds[1]
        ):
            break

    return scales


def detect_multiscale(
    detector: ACFDetector,
    image: np.ndarray,
    scales=None,
    stride=8,
    score_threshold=0.5,
    nms_threshold=0.3,
    batch_size=32,
    use_fast_pyramid=True,
) -> List[Tuple[int, int, int, int, float]]:
    if use_fast_pyramid:
        return detect_multiscale_fast(
            detector, image, scales, stride, score_threshold, nms_threshold, batch_size
        )

    if not detector.trained:
        raise ValueError("Detector must be trained before inference")

    if scales is None:
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]

    all_boxes = []
    all_scores = []

    pyramid = compute_channel_pyramid(image, scales)

    detector.classifier.eval()

    all_windows_data = []

    for scaled_img, scale in pyramid:
        h, w = scaled_img.shape[:2]
        win_w, win_h = detector.window_size

        if h < win_h or w < win_w:
            continue

        windows = generate_sliding_windows((h, w), detector.window_size, stride)

        for window in windows:
            all_windows_data.append((scaled_img, window, scale))

    for i in tqdm(
        range(0, len(all_windows_data), batch_size),
        desc="Detecting faces",
        unit="batch",
        ncols=100,
    ):
        batch_windows = all_windows_data[i : i + batch_size]
        batch_features = []
        batch_metadata = []

        for scaled_img, window, scale in batch_windows:
            try:
                features = detector.extract_features(scaled_img, window)
                batch_features.append(features)
                batch_metadata.append((window, scale))
            except Exception as e:
                print(f"[ERROR] {e}")
                continue

        if len(batch_features) == 0:
            continue

        features_numpy = np.array(batch_features)
        features_numpy_reshaped = features_numpy.reshape(
            -1, detector.feature_resolution, detector.feature_resolution, 10
        )
        features_normalized = (features_numpy_reshaped - detector.mean) / detector.std
        features_numpy = features_normalized.reshape(
            -1, detector.feature_resolution * detector.feature_resolution * 10
        )

        scores = detector.classifier.infer_batch(features_numpy)

        for idx, (window, scale) in enumerate(batch_metadata):
            score = scores[idx]

            if score > score_threshold:
                x, y, win_w, win_h = window

                orig_x = int(x / scale)
                orig_y = int(y / scale)
                orig_w = int(win_w / scale)
                orig_h = int(win_h / scale)

                all_boxes.append([orig_x, orig_y, orig_w, orig_h])
                all_scores.append(float(score))

    if len(all_boxes) > 0:
        keep_indices = non_max_suppression(all_boxes, all_scores, nms_threshold)

        detections = []
        for idx in keep_indices:
            box = all_boxes[idx]
            score = all_scores[idx]
            detections.append((*box, score))

        return detections

    return []


def visualize_detections(
    image: np.ndarray, detections: List[Tuple[int, int, int, int, float]], thickness=2
) -> np.ndarray:
    vis_image = image.copy()

    for detection in detections:
        x, y, w, h, score = detection

        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), thickness)

        text = f"{score:.2f}"
        cv2.putText(
            vis_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    return vis_image


def normalize_channel(channel):
    """Normalize a channel to 0-255 for visualization."""
    ch_min = channel.min()
    ch_max = channel.max()

    if ch_max - ch_min < 1e-6:
        return np.zeros_like(channel, dtype=np.uint8)

    normalized = (channel - ch_min) / (ch_max - ch_min) * 255
    return normalized.astype(np.uint8)


def visualize_feature_map(feature_map, detection_idx, score, original_image):
    channels_names = ["L", "U", "V", "M", "H1", "H2", "H3", "H4", "H5", "H6"]

    scale_factor = 8
    channel_size = feature_map.shape[0]
    upscaled_size = channel_size * scale_factor

    n_cols = 5
    n_rows = 2
    gap_between_rows = 32
    top_margin = 220
    left_margin = 20

    fig_height = top_margin + n_rows * upscaled_size + gap_between_rows
    fig_width = left_margin + n_cols * upscaled_size + 20

    figure = np.ones((fig_height, fig_width, 3), dtype=np.uint8) * 255

    title = f"Detection {detection_idx} (score: {score:.3f})"
    cv2.putText(figure, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    if original_image is not None:
        print(original_image.ndim)
        if original_image.ndim == 2:
            colored_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        else:
            colored_image = original_image

        max_w = 160
        print(colored_image.shape)
        h, w, _ = colored_image.shape
        scale = max_w / w
        resized = cv2.resize(
            colored_image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

        y0 = 32
        x0 = fig_width - resized.shape[1] - 24
        figure[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1], :] = resized

        cv2.putText(
            figure,
            "Original",
            (x0, y0 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    for idx, (channel, name) in enumerate(
        zip(feature_map.transpose(2, 0, 1), channels_names)
    ):
        row = idx // n_cols
        col = idx % n_cols

        y_start = top_margin + row * upscaled_size + row * gap_between_rows
        x_start = left_margin + col * upscaled_size

        normalized = normalize_channel(channel)
        upscaled = cv2.resize(
            normalized, (upscaled_size, upscaled_size), interpolation=cv2.INTER_NEAREST
        )
        upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2RGB)

        figure[
            y_start : y_start + upscaled_size, x_start : x_start + upscaled_size, :
        ] = upscaled_bgr

        cv2.putText(
            figure,
            name,
            (x_start, y_start - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

    return figure


def evaluate_detections(
    detections: List[Tuple[int, int, int, int, float]],
    ground_truth: List[Tuple[int, int, int, int]],
    iou_threshold=0.5,
) -> Dict[str, float]:
    total_predictions = len(detections)

    if len(ground_truth) == 0:
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "true_positives": 0,
            "false_positives": total_predictions,
            "false_negatives": 0,
            "total_predictions": total_predictions,
        }

    if len(detections) == 0:
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": total_predictions,
            "total_predictions": total_predictions,
        }

    matched_gt = set()
    true_positives = 0

    for det in detections:
        det_box = det[:4]
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue

            iou = compute_iou(det_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            true_positives += 1

    false_positives = total_predictions - true_positives
    false_negatives = len(ground_truth) - true_positives

    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    accuracy = true_positives / (
        true_positives + false_positives + false_negatives + 1e-6
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_predictions": total_predictions,
    }
