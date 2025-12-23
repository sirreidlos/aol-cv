from dataclasses import dataclass
import os
import numpy as np
import cv2
from typing import Dict, Literal, Optional, Tuple, List
import pickle
from tqdm.auto import tqdm

from acf.channels import compute_channels


@dataclass
class AnnotationSetting:
    filter_invalid: bool
    acceptable_blur: Optional[List[int]] = None
    acceptable_expression: Optional[List[int]] = None
    acceptable_illumination: Optional[List[int]] = None
    acceptable_occlusion: Optional[List[int]] = None
    acceptable_pose: Optional[List[int]] = None


def create_cache_key(
    dataset_type: Literal["widerface", "muct"],
    num_train_images: int,
    num_val_images: int,
    feature_resolution: int,
    window_size: Tuple[int, int],
    pos_iou_thresh: float,
    neg_iou_thresh: float,
    hard_neg_iou_range: Tuple[float, float],
    num_neg_per_pos: int,
) -> str:
    return (
        f"{dataset_type}_"
        f"train_{num_train_images}_val_{num_val_images}_"
        f"res_{feature_resolution}_win_{window_size[0]}x{window_size[1]}_"
        f"pos{pos_iou_thresh}_neg{neg_iou_thresh}_"
        f"hard{hard_neg_iou_range[0]}-{hard_neg_iou_range[1]}_"
        f"ratio{num_neg_per_pos}"
    )


def save_cache(
    cache_key: str,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    cache_dir: str = "cache",
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")

    with open(cache_file, "wb") as f:
        pickle.dump(
            {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
            },
            f,
        )

    print(f"Cached preprocessed features to {cache_file}")


def load_cache(
    cache_key: str,
    cache_dir: str = "cache",
):
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")

    if not os.path.exists(cache_file):
        return None

    print(f"Loading cached features from {cache_file}...")
    with open(cache_file, "rb") as f:
        return pickle.load(f)


def get_widerface_training_data(
    annotation_file: str,
    val_annotation_file: str,
    image_dir: str,
    val_image_dir: str,
    annotation_setting: AnnotationSetting,
    feature_resolution: int,
    hard_neg_iou_range: Tuple[float, float],
    max_images: Optional[int],
    max_val_images: Optional[int],
    neg_iou_thresh: float,
    num_neg_per_pos: int,
    pos_iou_thresh: float,
    window_size: Tuple[int, int],
):
    annotations = parse_wider_face_annotation(annotation_file, AnnotationSetting(True))
    image_paths = list(annotations.keys())
    if max_images:
        image_paths = image_paths[:max_images]

    val_annotations = parse_wider_face_annotation(
        val_annotation_file, annotation_setting
    )
    val_image_paths = list(val_annotations.keys())
    if max_val_images:
        val_image_paths = val_image_paths[:max_val_images]

    cache_key = create_cache_key(
        "widerface",
        len(image_paths),
        len(val_image_paths),
        feature_resolution,
        window_size,
        pos_iou_thresh,
        neg_iou_thresh,
        hard_neg_iou_range,
        num_neg_per_pos,
    )

    cached_data = load_cache(cache_key)
    if cached_data is not None:
        X_train = cached_data["X_train"]
        y_train = cached_data["y_train"]
        X_val = cached_data["X_val"]
        y_val = cached_data["y_val"]
        return X_train, y_train, X_val, y_val

    X_train, y_train = extract_dataset_samples(
        image_paths=image_paths,
        annotations=annotations,
        image_base_dir=image_dir,
        feature_resolution=feature_resolution,
        window_size=window_size,
        pos_iou_thresh=pos_iou_thresh,
        neg_iou_thresh=neg_iou_thresh,
        hard_neg_iou_range=hard_neg_iou_range,
        num_neg_per_pos=num_neg_per_pos,
        max_pos_multiplier=num_neg_per_pos,
        desc="Processing images",
    )

    X_val, y_val = extract_dataset_samples(
        image_paths=image_paths,
        annotations=val_annotations,
        image_base_dir=val_image_dir,
        feature_resolution=feature_resolution,
        window_size=window_size,
        pos_iou_thresh=pos_iou_thresh,
        neg_iou_thresh=neg_iou_thresh,
        hard_neg_iou_range=hard_neg_iou_range,
        num_neg_per_pos=num_neg_per_pos,
        max_pos_multiplier=num_neg_per_pos,
        desc="Processing images",
    )

    assert isinstance(X_val, np.ndarray)
    assert isinstance(y_val, np.ndarray)
    save_cache(cache_key, X_train, y_train, X_val, y_val)

    return X_train, y_train, X_val, y_val


def get_muct_annotations(
    annotation_file: str,
    image_dir: str,
    val_split: float = 0.2,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    annotations = parse_muct_annotation(annotation_file)
    valid_annotations = {}

    for img_path, annots in annotations.items():
        full_path = f"{image_dir}/{img_path}"
        img = cv2.imread(full_path)
        if img is not None:
            valid_annotations[img_path] = annots

    image_paths = sorted(valid_annotations.keys())
    n_val = int(len(image_paths) * val_split)

    train_image_paths = image_paths[n_val:]
    train_annotations = {img: valid_annotations[img] for img in train_image_paths}
    val_image_paths = image_paths[:n_val]
    val_annotations = {img: valid_annotations[img] for img in val_image_paths}

    return train_annotations, val_annotations


def get_muct_training_data(
    annotation_file: str,
    image_dir: str,
    feature_resolution: int,
    hard_neg_iou_range: Tuple[float, float],
    max_images: Optional[int],
    neg_iou_thresh: float,
    num_neg_per_pos: int,
    pos_iou_thresh: float,
    window_size: Tuple[int, int],
    val_split: float = 0.2,
):
    train_annotations, val_annotations = get_muct_annotations(
        annotation_file, image_dir, val_split=val_split
    )

    train_image_paths = list(train_annotations.keys())
    if max_images:
        train_image_paths = train_image_paths[:max_images]

    val_image_paths = list(val_annotations.keys())

    cache_key = create_cache_key(
        "muct",
        len(train_image_paths),
        len(val_image_paths),
        feature_resolution,
        window_size,
        pos_iou_thresh,
        neg_iou_thresh,
        hard_neg_iou_range,
        num_neg_per_pos,
    )

    cached_data = load_cache(cache_key)
    if cached_data is not None:
        X_train = cached_data["X_train"]
        y_train = cached_data["y_train"]
        X_val = cached_data["X_val"]
        y_val = cached_data["y_val"]
        return X_train, y_train, X_val, y_val

    X_train, y_train = extract_dataset_samples(
        image_paths=train_image_paths,
        annotations=train_annotations,
        image_base_dir=image_dir,
        feature_resolution=feature_resolution,
        window_size=window_size,
        pos_iou_thresh=pos_iou_thresh,
        neg_iou_thresh=neg_iou_thresh,
        hard_neg_iou_range=hard_neg_iou_range,
        num_neg_per_pos=num_neg_per_pos,
        max_pos_multiplier=num_neg_per_pos,
        desc="Processing train images",
    )

    X_val, y_val = extract_dataset_samples(
        image_paths=val_image_paths,
        annotations=val_annotations,
        image_base_dir=image_dir,
        feature_resolution=feature_resolution,
        window_size=window_size,
        pos_iou_thresh=pos_iou_thresh,
        neg_iou_thresh=neg_iou_thresh,
        hard_neg_iou_range=hard_neg_iou_range,
        num_neg_per_pos=num_neg_per_pos,
        max_pos_multiplier=num_neg_per_pos,
        desc="Processing val images",
    )

    assert isinstance(X_val, np.ndarray)
    assert isinstance(y_val, np.ndarray)

    save_cache(cache_key, X_train, y_train, X_val, y_val)
    return X_train, y_train, X_val, y_val


def extract_dataset_samples(
    image_paths,
    annotations,
    image_base_dir: str,
    feature_resolution,
    window_size: Tuple[int, int],
    pos_iou_thresh: float,
    neg_iou_thresh: float,
    hard_neg_iou_range: Tuple[float, float],
    num_neg_per_pos: int,
    max_pos_multiplier: int | None = None,
    desc: str = "Processing images",
):
    def extract_features(
        image: np.ndarray, roi: Tuple[int, int, int, int]
    ) -> np.ndarray:
        resized = resize_sample(image, roi, window_size)
        channels = compute_channels(resized)
        smoothed = cv2.GaussianBlur(channels, (3, 3), sigmaX=1)

        aggregated = cv2.resize(
            smoothed,
            (feature_resolution, feature_resolution),
            interpolation=cv2.INTER_AREA,
        )

        return aggregated.flatten()

    X_all, y_all = [], []

    for img_path in tqdm(image_paths, desc=desc, ncols=100):
        try:
            image = load_image(img_path, image_base_dir)
            gt_boxes = annotations[img_path]

            X, y = extract_samples_from_image(
                image=image,
                gt_boxes=gt_boxes,
                extract_features_fn=extract_features,
                window_size=window_size,
                pos_iou_thresh=pos_iou_thresh,
                neg_iou_thresh=neg_iou_thresh,
                hard_neg_iou_range=hard_neg_iou_range,
                num_neg_per_pos=num_neg_per_pos,
                max_pos_multiplier=max_pos_multiplier,
            )

            X_all.extend(X)
            y_all.extend(y)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    return (
        np.asarray(X_all, dtype=np.float32),
        np.asarray(y_all, dtype=np.int64),
    )


def extract_samples_from_image(
    image: np.ndarray,
    gt_boxes: np.ndarray,
    extract_features_fn,
    window_size: Tuple[int, int],
    pos_iou_thresh: float,
    neg_iou_thresh: float,
    hard_neg_iou_range: Tuple[float, float],
    num_neg_per_pos: int,
    scale: float = 1.0,
    max_pos_multiplier: int | None = None,
):
    h, w = image.shape[:2]
    win_w, win_h = window_size

    if h < win_h or w < win_w:
        return [], []

    pos_samples, neg_samples = extract_training_samples_sliding(
        image,
        gt_boxes,
        pos_iou_thresh=pos_iou_thresh,
        neg_iou_thresh=neg_iou_thresh,
        hard_neg_iou_range=hard_neg_iou_range,
        num_neg_per_pos=num_neg_per_pos,
        window_size=window_size,
        scale=scale,
    )

    X, y = [], []

    pos_count = 0
    neg_count = 0
    pos_limit = (
        len(gt_boxes) * max_pos_multiplier
        if max_pos_multiplier is not None
        else float("inf")
    )

    for box in pos_samples:
        if pos_count >= pos_limit:
            break
        X.append(extract_features_fn(image, box))
        y.append(1)
        pos_count += 1

    for box in neg_samples:
        if neg_count >= pos_count * num_neg_per_pos:
            break
        X.append(extract_features_fn(image, box))
        y.append(0)
        neg_count += 1

    return X, y


def parse_muct_annotation(annotation_file: str) -> Dict[str, np.ndarray]:
    annotations = {}

    with open(annotation_file, "r") as f:
        lines = f.readlines()

    i = 1
    while i < len(lines):
        boxes = []
        box_line = lines[i].strip().split(",")
        i += 1

        img_path = box_line[0] + ".jpg"

        values = list(map(float, box_line[2:]))
        assert len(values) == 152

        x_vals = values[0::2]
        y_vals = values[1::2]

        filtered_coords = [(x, y) for x, y in zip(x_vals, y_vals) if x != 0 and y != 0]
        if not filtered_coords:
            continue

        assert filtered_coords, "No valid landmarks"
        x_filtered, y_filtered = zip(*filtered_coords)

        xmin = min(x_filtered)
        xmax = max(x_filtered)
        ymin = min(y_filtered)
        ymax = max(y_filtered)

        x = int(xmin)
        y = int(ymin)
        w = int(xmax - xmin)
        h = int(ymax - ymin)

        if w > 0 and h > 0:
            boxes.append([x, y, w, h])

        if boxes:
            annotations[img_path] = np.array(boxes)

    return annotations


def parse_wider_face_annotation(
    annotation_file: str,
    annotation_setting: AnnotationSetting,
) -> Dict[str, np.ndarray]:
    annotations = {}

    def allowed_attribute(values: List[int]) -> bool:
        _, _, w, h = values[:4]
        blur, expression, illumination, invalid, occlusion, pose = values[4:10]

        if w <= 0 or h <= 0:
            return False

        if annotation_setting.filter_invalid and invalid == 1:
            return False

        if (
            annotation_setting.acceptable_blur is not None
            and blur not in annotation_setting.acceptable_blur
        ):
            return False

        if (
            annotation_setting.acceptable_expression is not None
            and expression not in annotation_setting.acceptable_expression
        ):
            return False

        if (
            annotation_setting.acceptable_illumination is not None
            and illumination not in annotation_setting.acceptable_illumination
        ):
            return False

        if (
            annotation_setting.acceptable_occlusion is not None
            and occlusion not in annotation_setting.acceptable_occlusion
        ):
            return False

        if (
            annotation_setting.acceptable_pose is not None
            and pose not in annotation_setting.acceptable_pose
        ):
            return False

        return True

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
            values = list(map(int, box_line[:10]))
            if not allowed_attribute(values):
                continue

            x, y, w, h = values[:4]
            if w > 0 and h > 0:
                boxes.append([x, y, w, h])

        if boxes:
            annotations[img_path] = np.array(boxes)

    return annotations


def compute_iou_batch(
    windows: List[Tuple[int, int, int, int]] | np.ndarray,
    boxes: List[Tuple[int, int, int, int]] | np.ndarray,
) -> np.ndarray:
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
) -> Tuple[np.ndarray, np.ndarray]:
    assert annotations.ndim == 2, f"Expected 2D annotations, got {annotations.ndim}D"
    assert annotations.shape[1] == 4, f"Expected shape [N, 4], got {annotations.shape}"
    height, width = image.shape[:2]

    pos_samples = []
    for gt_box in annotations:
        x, y, w, h = gt_box
        pos_samples.append([x, y, w, h])

    pos_samples = np.array(pos_samples)

    stride = min(window_size) // 4
    windows = generate_sliding_windows((height, width), window_size, stride)
    np.random.shuffle(windows)
    windows = np.array(windows)

    windows_scaled = windows / scale
    iou_matrix = compute_iou_batch(windows_scaled, annotations)
    max_ious = np.max(iou_matrix, axis=1)

    hard_mask = (max_ious >= hard_neg_iou_range[0]) & (
        max_ious <= hard_neg_iou_range[1]
    )
    neg_mask = max_ious <= neg_iou_thresh

    hard_neg_samples = windows[hard_mask]
    neg_samples = windows[neg_mask]

    num_neg_needed = len(pos_samples) * num_neg_per_pos
    num_hard = min(len(hard_neg_samples), num_neg_needed // 2)
    num_easy = min(len(neg_samples), num_neg_needed - num_hard)

    final_neg = np.vstack([hard_neg_samples[:num_hard], neg_samples[:num_easy]])

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
