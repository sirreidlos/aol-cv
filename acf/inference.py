from typing import Dict, List, Tuple
import numpy as np
import cv2
from tqdm import tqdm
import torch

from acf.model import ACFDetector
from .channels import compute_channel_pyramid
from .preprocessing import generate_sliding_windows


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


def detect_multiscale(
    detector: ACFDetector,
    image: np.ndarray,
    scales=None,
    stride=8,
    score_threshold=0.5,
    nms_threshold=0.3,
    batch_size=32,
) -> List[Tuple[int, int, int, int, float]]:
    """
    out: List of (x, y, w, h, score) tuples
    """

    if not detector.trained:
        raise ValueError("Detector must be trained before inference")

    if scales is None:
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]

    all_boxes = []
    all_scores = []

    pyramid = compute_channel_pyramid(image, scales)

    detector.classifier.eval()

    all_windows_data = []

    for scaled_img, _, scale in pyramid:
        h, w = scaled_img.shape[:2]
        win_w, win_h = detector.window_size

        if h < win_h or w < win_w:
            continue

        windows = generate_sliding_windows((h, w), detector.window_size, stride)

        for window in windows:
            all_windows_data.append((scaled_img, window, scale))

    with torch.no_grad():
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

            features_tensor = (
                torch.from_numpy(np.array(batch_features)).float().to(detector.device)
            )

            outputs = detector.classifier(features_tensor)
            probs = torch.softmax(outputs, dim=1)
            scores = probs[:, 1].cpu().numpy()

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


def evaluate_detections(
    detections: List[Tuple[int, int, int, int, float]],
    ground_truth: List[Tuple[int, int, int, int]],
    iou_threshold=0.5,
) -> Dict[str, float]:
    if len(ground_truth) == 0:
        return {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": len(ground_truth),
        }

    if len(detections) == 0:
        return {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "true_positives": 0,
            "false_positives": len(detections),
            "false_negatives": 0,
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

            iou = compute_iou_boxes(det_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            true_positives += 1

    false_positives = len(detections) - true_positives
    false_negatives = len(ground_truth) - true_positives

    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def compute_iou_boxes(
    box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
) -> float:
    x1, y1, w1, h1 = box1[:4]
    x2, y2, w2, h2 = box2[:4]

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)
