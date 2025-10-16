import os
import numpy as np
import cv2
from typing import Dict, Tuple, List


def parse_wider_face_annotation(
    annotation_file: str,
) -> Dict[str, np.ndarray]:
    annotations = {}

    with open(annotation_file, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        img_path = lines[i].strip()
        i += 1

        num_faces = int(lines[i].strip())
        i += 1

        boxes = []
        for _ in range(max(1, num_faces)):
            box_line = lines[i].strip().split()
            i += 1

            # WIDER FACE format: x, y, w, h, blur, expression, illumination, invalid, occlusion, pose
            x, y, w, h = map(int, box_line[:4])

            if w > 0 and h > 0:
                boxes.append([x, y, w, h])

        if boxes:
            annotations[img_path] = np.array(boxes)

    return annotations


def compute_iou_batch(windows: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    windows: [N;4]
    boxes: [M;4]
    out: [N, M]
    """
    windows = np.asarray(windows, dtype=np.float32).reshape(-1, 4)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    x1_w, y1_w, w_w, h_w = windows.T
    x2_w, y2_w = x1_w + w_w, y1_w + h_w
    x1_b, y1_b, w_b, h_b = boxes.T
    x2_b, y2_b = x1_b + w_b, y1_b + h_b

    inter_x1 = np.maximum(x1_w[:, None], x1_b[None, :])
    inter_y1 = np.maximum(y1_w[:, None], y1_b[None, :])
    inter_x2 = np.minimum(x2_w[:, None], x2_b[None, :])
    inter_y2 = np.minimum(y2_w[:, None], y2_b[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    inter_area = inter_w * inter_h

    area_w = (x2_w - x1_w) * (y2_w - y1_w)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)

    iou = inter_area / (area_w[:, None] + area_b[None, :] - inter_area + 1e-8)
    return iou


def compute_iou(
    box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
) -> float:
    """
    box1, box2: Bounding boxes in format [x, y, w, h]
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

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


def generate_sliding_windows(
    image_shape: Tuple[int, int], window_size: Tuple[int, int], stride: int
) -> List[Tuple[int, int, int, int]]:
    """
    NOTE:
    image_shape: (height, width) of the image
    window_size: (width, height) of the sliding window
    """
    height, width = image_shape
    win_w, win_h = window_size

    windows = []

    for y in range(0, height - win_h + 1, stride):
        for x in range(0, width - win_w + 1, stride):
            windows.append([x, y, win_w, win_h])

    return windows


def extract_training_samples_sliding(
    image: np.ndarray,
    annotations: np.ndarray,
    pos_iou_thresh: float,
    neg_iou_thresh: float,
    hard_neg_iou_range: Tuple[float, float],
    num_neg_per_pos: int,
    window_size: Tuple[int, int],
    scale: float,
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    assert annotations.ndim == 2, f"Expected 2D annotations, got {annotations.ndim}D"
    assert annotations.shape[1] == 4, f"Expected shape [N, 4], got {annotations.shape}"
    height, width = image.shape[:2]

    # pos_samples = []
    # neg_samples = []
    # hard_neg_samples = []

    stride = min(window_size) // 4
    windows = generate_sliding_windows((height, width), window_size, stride)
    np.random.shuffle(windows)
    windows = np.array(windows)

    windows_scaled = windows / scale
    # windows_scaled = np.asarray(windows_scaled, dtype=np.float32).reshape(-1, 4)
    # annotations = np.asarray(annotations, dtype=np.float32).reshape(-1, 4)
    iou_matrix = compute_iou_batch(windows_scaled, annotations)
    max_ious = np.max(iou_matrix, axis=1)

    pos_mask = max_ious >= pos_iou_thresh
    hard_mask = (max_ious >= hard_neg_iou_range[0]) & (max_ious < hard_neg_iou_range[1])
    neg_mask = max_ious < neg_iou_thresh

    pos_samples = windows[pos_mask]
    hard_neg_samples = windows[hard_mask]
    neg_samples = windows[neg_mask]

    # for win in windows:
    #     x, y, win_w, win_h = win

    #     orig_x, orig_y = int(x / scale), int(y / scale)
    #     orig_w, orig_h = int(win_w / scale), int(win_h / scale)

    #     orig_win = (orig_x, orig_y, orig_w, orig_h)

    #     max_iou = 0
    #     for gt_box in annotations:
    #         max_iou = max(max_iou, compute_iou(orig_win, gt_box))

    #     if max_iou >= pos_iou_thresh:
    #         pos_samples.append(win)
    #     elif hard_neg_iou_range[0] <= max_iou < hard_neg_iou_range[1]:
    #         hard_neg_samples.append(win)
    #     elif max_iou < neg_iou_thresh:
    #         neg_samples.append(win)

    num_neg_needed = len(pos_samples) * num_neg_per_pos
    num_hard = min(len(hard_neg_samples), num_neg_needed // 2)
    num_easy = min(len(neg_samples), num_neg_needed - num_hard)

    final_neg = np.vstack([hard_neg_samples[:num_hard], neg_samples[:num_easy]])
    # final_neg = hard_neg_samples[:num_hard] + neg_samples[:num_easy]

    return pos_samples, final_neg


def load_image(image_path: str, base_dir: str | None = None) -> np.ndarray:
    if base_dir:
        full_path = os.path.join(base_dir, image_path)
    else:
        full_path = image_path

    image = cv2.imread(full_path)
    if image is None:
        raise ValueError(f"Could not load image: {full_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def resize_sample(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
    target_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    x, y, w, h = roi
    patch = image[y : y + h, x : x + w]

    resized = cv2.resize(patch, target_size, interpolation=cv2.INTER_LINEAR)
    return resized
