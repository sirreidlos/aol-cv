from pathlib import Path
import pickle

from sklearn.metrics import classification_report
from acf.ada import AdaBoost
from acf.cnn import CNNClassifier
import numpy as np
from tqdm import tqdm
import os
from typing import Literal, Tuple, List

from acf.gbm import LightGBM
from acf.mlp import MLPClassifier
from acf.abstract_model import Model
from .channels import compute_channels
from .preprocessing import (
    AnnotationSetting,
    compute_iou_batch,
    extract_training_samples_sliding,
    generate_sliding_windows,
    parse_wider_face_annotation,
    load_image,
    resize_sample,
)
import torch


import gc
import heapq
from dataclasses import dataclass
import cv2

from concurrent.futures import ThreadPoolExecutor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ACFDetector:
    classifier: Model
    window_size: Tuple[int, int]
    feature_resolution: int

    def __init__(
        self,
        window_size=(64, 64),
        hidden_sizes=[512, 256],
        feature_resolution=16,
        learning_rate=0.001,
        batch_size=32,
        epochs=10,
        selection_metric="f_beta",
        pos_iou_thresh=0.5,
        neg_iou_thresh=0.3,
        hard_neg_iou_range=(0.1, 0.3),
        num_neg_per_pos=3,
        model: Literal["mlp", "cnn", "gbm", "ada"] = "mlp",
    ):
        self.window_size = window_size
        self.hidden_sizes = hidden_sizes
        self.feature_resolution = feature_resolution
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.selection_metric = selection_metric
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.hard_neg_iou_range = hard_neg_iou_range
        self.num_neg_per_pos = num_neg_per_pos

        input_size = 10 * feature_resolution * feature_resolution

        if model == "mlp":
            self.classifier = MLPClassifier(
                input_size=input_size, hidden_sizes=hidden_sizes, num_classes=2
            )
        elif model == "cnn":
            self.classifier = CNNClassifier(
                input_size,
                hidden_sizes,
                num_classes=2,
                feature_resolution=feature_resolution,
            )
        elif model == "gbm":
            self.classifier = LightGBM(
                n_estimators=100,
                learning_rate=1.0,
                max_depth=1,
                random_state=42,
                n_jobs=-1,
                verbose=1,
            )
        elif model == "ada":
            self.classifier = AdaBoost(
                n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42
            )

        self.model_type = model

        self.trained = False

    def extract_features_batch(self, channels_or_image, batch_windows):
        if len(channels_or_image.shape) == 3 and channels_or_image.shape[2] == 10:
            channels = channels_or_image
            with ThreadPoolExecutor(max_workers=8) as executor:
                features = list(
                    executor.map(
                        lambda w: self.extract_features_from_channels(channels, w),
                        batch_windows,
                    )
                )
        else:
            image = channels_or_image
            with ThreadPoolExecutor(max_workers=8) as executor:
                features = list(
                    executor.map(
                        lambda w: self.extract_features(image, w), batch_windows
                    )
                )
        return np.array(features, dtype=np.float32)

    def extract_features_from_channels(
        self, channels: np.ndarray, roi: Tuple[int, int, int, int]
    ) -> np.ndarray:
        x, y, w, h = roi
        x, y, w, h = int(x), int(y), int(w), int(h)

        patch = channels[y : y + h, x : x + w]
        resized_to_window = cv2.resize(
            patch, self.window_size, interpolation=cv2.INTER_LINEAR
        )
        smoothed = cv2.GaussianBlur(resized_to_window, (3, 3), sigmaX=1)

        aggregated = cv2.resize(
            smoothed,
            (self.feature_resolution, self.feature_resolution),
            interpolation=cv2.INTER_AREA,
        )

        return aggregated.flatten()

    def extract_features(
        self, image: np.ndarray, roi: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        image: [H, W, 3] (RGB)
        roi: [x, y, w, h]
        out: [self.feature_resolution * self.feature_resolution * 10]
        """
        resized = resize_sample(image, roi, self.window_size)
        channels = compute_channels(resized)
        smoothed = cv2.GaussianBlur(channels, (3, 3), sigmaX=1)

        aggregated = cv2.resize(
            smoothed,
            (self.feature_resolution, self.feature_resolution),
            interpolation=cv2.INTER_AREA,
        )

        return aggregated.flatten()

    def _get_cache_key(
        self,
        num_train_images: int,
        num_val_images: int,
        feature_resolution: int,
        window_size: Tuple[int, int],
    ):
        key_str = (
            f"train_{num_train_images}_val_{num_val_images}_"
            f"res_{feature_resolution}_win_{window_size[0]}x{window_size[1]}_"
            f"pos{self.pos_iou_thresh}_neg{self.neg_iou_thresh}_"
            f"hard{self.hard_neg_iou_range[0]}-{self.hard_neg_iou_range[1]}_"
            f"ratio{self.num_neg_per_pos}"
        )
        return key_str

    def _save_cache(self, cache_key: str, X_train, y_train, X_val=None, y_val=None):
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")

        cache_data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
        }

        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        print(f"Cached preprocessed features to {cache_file}")

    def _load_cache(self, cache_key: str):
        cache_dir = "cache"
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")

        if os.path.exists(cache_file):
            print(f"Loading cached features from {cache_file}...")
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
            return cache_data
        return None

    def get_train_data(
        self,
        annotation_setting: AnnotationSetting,
        annotation_file: str = "data/wider_face_split/wider_face_train_bbx_gt.txt",
        image_base_dir: str = "data/WIDER_train/images/",
        max_images=None,
        val_annotation_file=None,
        val_image_base_dir=None,
        max_val_images=None,
    ):
        """
        returns X_train, y_train, X_val, y_val
        """

        print("Loading annotations...")
        annotations = parse_wider_face_annotation(annotation_file, annotation_setting)

        image_paths = list(annotations.keys())
        if max_images:
            image_paths = image_paths[:max_images]

        num_val_images = 0
        if val_annotation_file and val_image_base_dir:
            val_annotation_setting = AnnotationSetting(
                acceptable_blur=None,
                acceptable_expression=None,
                acceptable_illumination=None,
                acceptable_occlusion=None,
                acceptable_pose=None,
                filter_invalid=True,
            )
            val_annotations = parse_wider_face_annotation(
                val_annotation_file, val_annotation_setting
            )  # have val set use all the images
            val_image_paths = list(val_annotations.keys())
            if max_val_images:
                val_image_paths = val_image_paths[:max_val_images]
            num_val_images = len(val_image_paths)

        cache_key = self._get_cache_key(
            len(image_paths), num_val_images, self.feature_resolution, self.window_size
        )

        # try loading from cache
        cache_data = self._load_cache(cache_key)

        if cache_data is not None:
            print("Using cached preprocessed features!")
            X_train = cache_data["X_train"]
            y_train = cache_data["y_train"]
            X_val = cache_data["X_val"]
            y_val = cache_data["y_val"]

            print(f"Loaded {len(X_train)} training samples from cache")
            print(f"Positive samples: {np.sum(y_train == 1)}")
            print(f"Negative samples: {np.sum(y_train == 0)}")

            if X_val is not None:
                print(f"Loaded {len(X_val)} validation samples from cache")
                print(f"Positive validation samples: {np.sum(y_val == 1)}")
                print(f"Negative validation samples: {np.sum(y_val == 0)}")

            return X_train, y_train, X_val, y_val

        # no cache artifacts
        print(f"Cache miss - extracting features from {len(image_paths)} images...")

        X_train = []
        y_train = []

        for img_path in tqdm(image_paths, desc="Processing images", ncols=100):
            try:
                image = load_image(img_path, image_base_dir)
                gt_boxes = annotations[img_path]

                pos_count = 0
                neg_count = 0
                pos_samples_needed = len(gt_boxes) * self.num_neg_per_pos

                h, w = image.shape[:2]
                win_w, win_h = self.window_size

                if h >= win_h and w >= win_w:
                    pos_samples, neg_samples = extract_training_samples_sliding(
                        image,
                        gt_boxes,
                        pos_iou_thresh=self.pos_iou_thresh,
                        neg_iou_thresh=self.neg_iou_thresh,
                        hard_neg_iou_range=self.hard_neg_iou_range,
                        num_neg_per_pos=self.num_neg_per_pos,
                        window_size=self.window_size,
                        scale=1.0,
                    )

                    for pos_sample in pos_samples:
                        if pos_count >= pos_samples_needed:
                            break

                        features = self.extract_features(image, pos_sample)
                        X_train.append(features)
                        y_train.append(1)
                        pos_count += 1

                    for neg_sample in neg_samples:
                        if neg_count >= pos_count * self.num_neg_per_pos:
                            break

                        features = self.extract_features(image, neg_sample)
                        X_train.append(features)
                        y_train.append(0)
                        neg_count += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int64)

        print(f"\nTraining classifier on {len(X_train)} samples...")
        print(f"Positive samples: {np.sum(y_train == 1)}")
        print(f"Negative samples: {np.sum(y_train == 0)}")

        X_val, y_val = None, None
        if not val_annotation_file or not val_image_base_dir:
            self._save_cache(cache_key, X_train, y_train, X_val, y_val)
            return X_train, y_train, X_val, y_val

        print("\nLoading validation data...")
        val_annotations = parse_wider_face_annotation(
            val_annotation_file, annotation_setting
        )

        print(f"Extracting features from {len(val_image_paths)} validation images...")

        X_val_list = []
        y_val_list = []

        for img_path in tqdm(
            val_image_paths, desc="Processing validation images", ncols=100
        ):
            try:
                image = load_image(img_path, val_image_base_dir)
                gt_boxes = val_annotations[img_path]

                pos_count = 0
                neg_count = 0
                pos_samples_needed = len(gt_boxes) * 3

                h, w = image.shape[:2]
                win_w, win_h = self.window_size

                if h >= win_h and w >= win_w:
                    pos_samples, neg_samples = extract_training_samples_sliding(
                        image,
                        gt_boxes,
                        pos_iou_thresh=self.pos_iou_thresh,
                        neg_iou_thresh=self.neg_iou_thresh,
                        hard_neg_iou_range=self.hard_neg_iou_range,
                        num_neg_per_pos=self.num_neg_per_pos,
                        window_size=self.window_size,
                        scale=1.0,
                    )

                    for pos_sample in pos_samples:
                        if pos_count >= pos_samples_needed:
                            break

                        features = self.extract_features(image, pos_sample)
                        X_val_list.append(features)
                        y_val_list.append(1)
                        pos_count += 1

                    for neg_sample in neg_samples:
                        if neg_count >= pos_count * self.num_neg_per_pos:
                            break

                        features = self.extract_features(image, neg_sample)
                        X_val_list.append(features)
                        y_val_list.append(0)
                        neg_count += 1

            except Exception as e:
                print(f"Error processing validation image {img_path}: {e}")
                continue

        X_val = np.array(X_val_list, dtype=np.float32)
        y_val = np.array(y_val_list, dtype=np.int64)

        print(f"Validation samples: {len(X_val)}")
        print(f"Positive validation samples: {np.sum(y_val == 1)}")
        print(f"Negative validation samples: {np.sum(y_val == 0)}")

        self._save_cache(cache_key, X_train, y_train, X_val, y_val)
        return X_train, y_train, X_val, y_val

    def save(self, filepath: str):
        model_data = {
            "model_state": self.classifier.get_state(),
            "window_size": self.window_size,
            "hidden_sizes": self.hidden_sizes,
            "feature_resolution": self.feature_resolution,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "trained": self.trained,
            "std": self.std,
            "mean": self.mean,
            "model_type": self.model_type,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load(self, filepath: str | Path):
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.window_size = model_data["window_size"]
        self.hidden_sizes = model_data["hidden_sizes"]
        self.feature_resolution = model_data.get("feature_resolution")
        self.learning_rate = model_data.get("learning_rate")
        self.batch_size = model_data.get("batch_size")
        self.epochs = model_data.get("epochs")
        self.trained = model_data["trained"]
        self.mean = model_data["mean"]
        self.std = model_data["std"]
        self.model_type = model_data.get("model_type", "mlp")

        input_size = 10 * self.feature_resolution * self.feature_resolution

        if self.model_type == "mlp":
            self.classifier = MLPClassifier(
                input_size=input_size, hidden_sizes=self.hidden_sizes, num_classes=2
            ).to(DEVICE)
        elif self.model_type == "cnn":
            self.classifier = CNNClassifier(
                input_size,
                self.hidden_sizes,
                num_classes=2,
                feature_resolution=self.feature_resolution,
            ).to(DEVICE)
        elif self.model_type == "gbm":
            self.classifier = LightGBM(
                n_estimators=100,
                learning_rate=1.0,
                max_depth=1,
                random_state=42,
                n_jobs=-1,
                verbose=1,
            )
        elif self.model_type == "ada":
            self.classifier = AdaBoost(
                n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42
            )

        self.classifier.load_state(model_data["model_state"])
        self.classifier.eval()

        print(f"Model loaded from {filepath}")

    def report(self, X_val, y_val):
        """Legacy patch-based classification report (for comparison only)"""
        print("=== LEGACY PATCH-BASED CLASSIFICATION REPORT ===")
        pred_probs = self.classifier.infer_batch(X_val)
        pred_labels = (pred_probs >= 0.5).astype(int)
        print(classification_report(y_val, pred_labels))
        print("WARNING: This does NOT reflect real detection performance!")
        print("Use evaluate_detection() for proper object detection metrics.\n")

    def normalize_in_place(self, X: np.ndarray) -> np.ndarray:
        """Normalize features using pre-computed mean and std"""
        X = X.reshape(-1, self.feature_resolution, self.feature_resolution, 10)
        X = (X - self.mean) / self.std
        return X.reshape(-1, self.feature_resolution * self.feature_resolution * 10)

    def evaluate_detection(
        self,
        annotation_file,
        image_base_dir,
        iou_threshold=0.5,
        confidence_threshold=0.5,
        max_images=None,
    ):
        """
        Proper object detection evaluation using sliding window detection
        on full images with IoU-based matching.
        """
        from acf.preprocessing import parse_wider_face_annotation
        from acf.channels import compute_channels
        from .preprocessing import (
            generate_sliding_windows,
            compute_iou_batch,
            load_image,
        )

        print("=== PROPER OBJECT DETECTION EVALUATION ===")

        # Load annotations
        annotation_setting = AnnotationSetting(
            acceptable_blur=None,
            acceptable_expression=None,
            acceptable_illumination=None,
            acceptable_occlusion=None,
            acceptable_pose=None,
            filter_invalid=True,
        )
        annotations = parse_wider_face_annotation(annotation_file, annotation_setting)
        image_paths = list(annotations.keys())

        if max_images:
            image_paths = image_paths[:max_images]

        print(f"Evaluating on {len(image_paths)} images...")

        total_gt_boxes = 0
        total_detections = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for img_path in tqdm(image_paths, desc="Detecting faces", ncols=100):
            try:
                # Load image
                image = load_image(img_path, image_base_dir)
                gt_boxes = annotations[img_path]
                h, w = image.shape[:2]

                # Skip if image too small
                if h < self.window_size[1] or w < self.window_size[0]:
                    continue

                # Generate sliding windows
                stride = max(4, min(self.window_size) // 4)
                windows = generate_sliding_windows((h, w), self.window_size, stride)
                windows = np.array(windows)  # Convert to numpy array

                # Debug: Check shapes and types
                print(f"Debug: windows shape: {windows.shape}, dtype: {windows.dtype}")

                # Extract features for all windows
                channels = compute_channels(image)
                features = self.extract_features_batch(channels, windows)
                features_norm = self.normalize_in_place(features.copy())

                # Get predictions
                scores = self.classifier.infer_batch(features_norm)

                # Filter detections by confidence threshold
                confident_mask = scores >= confidence_threshold
                confident_detections = windows[confident_mask]
                confident_scores = scores[confident_mask]

                # Non-maximum suppression (simple version)
                if len(confident_detections) > 0:
                    # Sort by confidence
                    sorted_indices = np.argsort(confident_scores)[::-1]
                    confident_detections = confident_detections[sorted_indices]

                    # Simple NMS
                    final_detections = []
                    for det in confident_detections:
                        keep = True
                        for kept_det in final_detections:
                            iou = compute_iou_batch(
                                np.array([det]), np.array([kept_det])
                            )[0, 0]
                            if iou > 0.3:  # NMS threshold
                                keep = False
                                break
                        if keep:
                            final_detections.append(det)

                    confident_detections = (
                        np.array(final_detections)
                        if final_detections
                        else np.array([]).reshape(0, 4)
                    )
                else:
                    confident_detections = np.array([]).reshape(0, 4)

                # Match detections to ground truth
                if len(gt_boxes) > 0 and len(confident_detections) > 0:
                    iou_matrix = compute_iou_batch(confident_detections, gt_boxes)

                    # Greedy matching
                    matched_gt = set()
                    tp_count = 0

                    for det_idx in range(len(confident_detections)):
                        best_gt_idx = np.argmax(iou_matrix[det_idx])
                        best_iou = iou_matrix[det_idx, best_gt_idx]

                        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                            tp_count += 1
                            matched_gt.add(best_gt_idx)

                    true_positives += tp_count
                    false_positives += len(confident_detections) - tp_count
                    false_negatives += len(gt_boxes) - len(matched_gt)

                elif len(gt_boxes) == 0:
                    false_positives += len(confident_detections)
                else:  # len(confident_detections) == 0
                    false_negatives += len(gt_boxes)

                total_gt_boxes += len(gt_boxes)
                total_detections += len(confident_detections)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        # Calculate metrics
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print("\nDetection Results:")
        print(f"  Ground truth faces: {total_gt_boxes}")
        print(f"  Total detections: {total_detections}")
        print(f"  True positives: {true_positives}")
        print(f"  False positives: {false_positives}")
        print(f"  False negatives: {false_negatives}")
        print(f"\nMetrics (IoU threshold: {iou_threshold}):")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": true_positives,
            "fp": false_positives,
            "fn": false_negatives,
            "total_gt": total_gt_boxes,
            "total_detections": total_detections,
        }


@dataclass
class HardNegative:
    score: float
    features: np.ndarray

    def __lt__(self, other):
        return self.score < other.score

    def __le__(self, other):
        return self.score <= other.score


class HeapBasedHardNegativeMiner:
    def __init__(self, max_size: int, feature_size: int):
        self.max_size = max_size
        self.feature_size = feature_size
        self.main_heap: List[HardNegative] = []
        self.candidate_heap: List[HardNegative] = []

    def add(self, score: float, features_normalized: np.ndarray):
        """
        score: confidence score from model
        features_normalized: normalized features
        """
        if len(self.candidate_heap) < self.max_size:
            heapq.heappush(
                self.candidate_heap, HardNegative(score, features_normalized.copy())
            )
        elif score > self.candidate_heap[0].score:
            heapq.heapreplace(
                self.candidate_heap, HardNegative(score, features_normalized.copy())
            )

    def add_batch(self, scores: np.ndarray, features_normalized: np.ndarray):
        sorted_indices = np.argsort(scores)[::-1]

        for idx in sorted_indices:
            score = float(scores[idx])
            if (
                len(self.candidate_heap) >= self.max_size
                and score <= self.candidate_heap[0].score
            ):
                break
            self.add(score, features_normalized[idx])

    def rescore_and_merge(self, detector, batch_size: int = 1000):
        print(f"  Re-scoring {len(self.main_heap)} existing negatives...")

        stats = {
            "old_count": len(self.main_heap),
            "new_candidates": len(self.candidate_heap),
            "dropped": 0,
            "retained": 0,
        }

        old_scores = []

        if len(self.main_heap) > 0:
            old_scores = [item.score for item in self.main_heap]
            all_features = np.vstack([item.features for item in self.main_heap])

            new_scores = []
            for i in range(0, len(all_features), batch_size):
                batch = all_features[i : i + batch_size]
                scores = detector.classifier.infer_batch(batch)
                new_scores.extend(scores)

            new_scores = np.array(new_scores)

            avg_decrease = np.mean(
                [old - new for old, new in zip(old_scores, new_scores)]
            )

            print(f"    └ Average score decrease: {avg_decrease:.4f}")

            self.main_heap = [
                HardNegative(float(new_scores[i]), item.features)
                for i, item in enumerate(self.main_heap)
            ]
            heapq.heapify(self.main_heap)

        print(f"  Merging {len(self.candidate_heap)} new candidates...")
        combined = self.main_heap + self.candidate_heap
        combined.sort(key=lambda x: x.score, reverse=True)

        if len(combined) > self.max_size:
            dropped_negatives = len(combined) - self.max_size
            stats["dropped"] = dropped_negatives
            print(f"    └ Dropping {dropped_negatives} negatives that are too easy")

        self.main_heap = combined[: self.max_size]
        heapq.heapify(self.main_heap)

        stats["retained"] = len(self.main_heap)

        self.candidate_heap = []

        if len(self.main_heap) > 0:
            max_score = max(item.score for item in self.main_heap)
            mean_score = np.mean([item.score for item in self.main_heap])
            print(f"  ✓ Final heap: {len(self.main_heap)} negatives")
            print(
                f"    Score range: [{self.main_heap[0].score:.4f}, {max_score:.4f}], mean: {mean_score:.4f}"
            )

        gc.collect()
        return stats

    def normalize_features(self, detector, X: np.ndarray) -> np.ndarray:
        X = X.reshape(-1, detector.feature_resolution, detector.feature_resolution, 10)
        X = (X - detector.mean) / detector.std
        return X.reshape(
            -1, detector.feature_resolution * detector.feature_resolution * 10
        )

    def get_all_features(self) -> np.ndarray:
        if len(self.main_heap) == 0:
            return np.array([]).reshape(0, self.feature_size)

        features = np.vstack([item.features for item in self.main_heap])
        return features.astype(np.float32)

    def get_score_stats(self) -> dict:
        if len(self.main_heap) == 0:
            return {
                "count": 0,
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
                "candidates": len(self.candidate_heap),
            }

        scores = [item.score for item in self.main_heap]
        min_scores = min(scores)
        assert min_scores == self.main_heap[0].score, (
            f"{min_scores} != {self.main_heap[0].score}"
        )
        return {
            "count": len(scores),
            "min": min(scores),
            "max": max(scores),
            "mean": np.mean(scores),
            "median": np.median(scores),
            "candidates": len(self.candidate_heap),
        }

    def __len__(self):
        return len(self.main_heap)


class MemoryEfficientBootstrapWithHeap:
    """
    Bootstrap training with heap-based hard negative mining.

    Benefits over circular buffer:
    - Always keeps the HARDEST negatives, not just recent ones
    - Automatically adapts as model improves
    - Same memory footprint, better quality
    - Only O(log K) overhead per insertion
    """

    detector: ACFDetector

    def __init__(self, detector: ACFDetector):
        self.detector = detector

    def get_neg_current(
        self,
        round_idx: int,
        X_pos,
        y_pos,
        X_neg_initial,
        y_neg_initial,
        mining_annotation_file,
        mining_image_base_dir,
        hard_neg_miner,
        num_mining_images,
        annotation_setting: AnnotationSetting,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if round_idx == 0:
            n_neg = min(len(X_neg_initial), len(X_pos) * 3)
            neg_indices = np.random.choice(len(X_neg_initial), n_neg, replace=False)

            X_neg_current = self.normalize_chunks(X_neg_initial[neg_indices])
            y_neg_current = y_neg_initial[neg_indices].copy()

            print(f"Round 0: Using {n_neg} initial random negatives")
            return (X_neg_current, y_neg_current)

        if not mining_annotation_file or not mining_image_base_dir:
            print("  Warning: No mining config provided")
            n_neg = len(X_pos) * 3
            neg_indices = np.random.choice(len(X_neg_initial), n_neg, replace=False)
            X_neg_current = self.normalize_chunks(X_neg_initial[neg_indices])
            y_neg_current = y_neg_initial[neg_indices]
            return (X_neg_current, y_neg_current)

        # score_threshold = round_idx * score_stride
        score_threshold = 0.5

        self.mine_hard_negatives_to_heap(
            hard_neg_miner,
            mining_annotation_file,
            mining_image_base_dir,
            annotation_setting=annotation_setting,
            num_images=num_mining_images,
            max_hard_negs_per_image=50,
            score_threshold=score_threshold,
        )

        stats = hard_neg_miner.get_score_stats()
        print("\n  Before merge:")
        print(f"  ├ Main heap: {stats['count']} hard negatives")
        print(f"  └ New candidates: {stats['candidates']}")

        if stats["count"] > 0:
            print(
                f"      Main heap score range: [{stats['min']:.4f}, {stats['max']:.4f}]"
            )

        if stats["count"] > 0:
            print("\n  🔄 Re-scoring existing negatives with current model...")
            hard_neg_miner.rescore_and_merge(self.detector, batch_size=1000)
        else:
            print("\n  First mining round: promoting candidates to main heap...")
            hard_neg_miner.main_heap = hard_neg_miner.candidate_heap
            hard_neg_miner.candidate_heap = []
            heapq.heapify(hard_neg_miner.main_heap)
            if len(hard_neg_miner.main_heap) > 0:
                max_score = max(item.score for item in hard_neg_miner.main_heap)
                mean_score = np.mean([item.score for item in hard_neg_miner.main_heap])
                print(
                    f"      ✓ Promoted {len(hard_neg_miner.main_heap)} negatives to main heap"
                )
                print(
                    f"      Score range: [{hard_neg_miner.main_heap[0].score:.4f}, {max_score:.4f}], mean: {mean_score:.4f}"
                )

        if len(hard_neg_miner) > 0:
            # Get all hard negatives from heap
            X_all_hard = hard_neg_miner.get_all_features()

            # Mix with some initial negatives
            n_initial = min(len(X_neg_initial), len(X_pos))
            init_indices = np.random.choice(
                len(X_neg_initial), n_initial, replace=False
            )
            X_initial_norm = self.normalize_chunks(X_neg_initial[init_indices])

            X_neg_current = np.vstack([X_all_hard, X_initial_norm])
            y_neg_current = np.zeros(len(X_neg_current), dtype=np.int64)

            del X_initial_norm
            gc.collect()

            print(
                f"  Training with {len(X_all_hard)} mined + {n_initial} initial negatives"
            )
            print(f"  Pos:Neg ratio: 1:{len(X_neg_current) / len(X_pos):.1f}")
        else:
            print("  Warning: No hard negatives found, using initial negatives")
            n_neg = len(X_pos) * 3
            neg_indices = np.random.choice(len(X_neg_initial), n_neg, replace=False)
            X_neg_current = self.normalize_chunks(X_neg_initial[neg_indices])
            y_neg_current = y_neg_initial[neg_indices]

        return (X_neg_current, y_neg_current)

    def train_with_bootstrap(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        annotation_setting: AnnotationSetting,
        bootstrap_rounds=3,
        early_stopping_patience=5,
        mining_annotation_file: str | None = None,
        mining_image_base_dir: str | None = None,
        num_mining_images: int = 100,
        max_hard_neg_buffer_size: int = 100000,
    ):
        print(f"Any NaN: {np.isnan(X_train).any()}")
        print(f"Any Inf: {np.isinf(X_train).any()}")

        pos_mask = y_train == 1
        neg_mask = y_train == 0

        X_pos = X_train[pos_mask].copy()
        y_pos = y_train[pos_mask].copy()
        X_neg_initial = X_train[neg_mask].copy()
        y_neg_initial = y_train[neg_mask].copy()

        del X_train, y_train, pos_mask, neg_mask
        gc.collect()

        print("Memory freed after splitting train data")

        print("\nBootstrap Training Setup:")
        print(f"  ├ Positive samples: {len(X_pos)}")
        print(f"  ├ Initial negatives: {len(X_neg_initial)}")
        print(f"  └ Bootstrap rounds: {bootstrap_rounds}")

        sample_size = min(10000, len(X_pos) + len(X_neg_initial))
        sample_indices_pos = np.random.choice(
            len(X_pos), min(sample_size // 2, len(X_pos)), replace=False
        )
        sample_indices_neg = np.random.choice(
            len(X_neg_initial), min(sample_size // 2, len(X_neg_initial)), replace=False
        )

        X_sample = np.vstack(
            [X_pos[sample_indices_pos], X_neg_initial[sample_indices_neg]]
        )
        X_sample_r = X_sample.reshape(
            -1, self.detector.feature_resolution, self.detector.feature_resolution, 10
        )

        mean = X_sample_r.mean(axis=(0, 1, 2), keepdims=True)
        std = X_sample_r.std(axis=(0, 1, 2), keepdims=True) + 1e-8

        self.detector.mean = mean.astype(np.float32)
        self.detector.std = std.astype(np.float32)

        del X_sample, X_sample_r
        gc.collect()

        print("\nNormalization stats computed from sample")

        print(f"Normalizing {len(X_pos)} positive samples...")
        X_pos = self.normalize_in_place(X_pos)
        gc.collect()

        feature_size = X_pos.shape[1]
        hard_neg_miner = HeapBasedHardNegativeMiner(
            max_hard_neg_buffer_size, feature_size
        )
        print(f"Initialized heap-based miner (capacity: {max_hard_neg_buffer_size})")

        if X_val is not None:
            print(f"Normalizing {len(X_val)} validation samples...")
            X_val = self.normalize_in_place(X_val)
            gc.collect()

        best_overall_metric = (
            0.0 if self.detector.selection_metric != "val_loss" else float("inf")
        )
        best_overall_state = None

        for round_idx in range(bootstrap_rounds + 1):
            print(f"\n{'=' * 60}")
            print(f"Bootstrap Round {round_idx}/{bootstrap_rounds}")
            print(f"{'=' * 60}")

            X_neg_current, y_neg_current = self.get_neg_current(
                round_idx=round_idx,
                X_pos=X_pos,
                y_pos=y_pos,
                X_neg_initial=X_neg_initial,
                y_neg_initial=y_neg_initial,
                mining_annotation_file=mining_annotation_file,
                mining_image_base_dir=mining_image_base_dir,
                hard_neg_miner=hard_neg_miner,
                num_mining_images=num_mining_images,
                annotation_setting=annotation_setting,
            )

            X_current = np.vstack([X_pos, X_neg_current])
            y_current = np.hstack([y_pos, y_neg_current])

            del X_neg_current, y_neg_current
            gc.collect()

            shuffle_idx = np.random.permutation(len(X_current))
            X_current = X_current[shuffle_idx]
            y_current = y_current[shuffle_idx]

            print(
                f"Training on {len(X_current)} samples (pos: {np.sum(y_current == 1)}, neg: {np.sum(y_current == 0)})"
            )

            if round_idx > 0:
                print("Continuing from previous round's weights...")

            self.detector.classifier.train_init(
                X_current,
                y_current,
                X_val,
                y_val,
                epochs=self.detector.epochs,
                batch_size=self.detector.batch_size,
                learning_rate=self.detector.learning_rate,
            )

            best_metric = (
                0.0 if self.detector.selection_metric != "val_loss" else float("inf")
            )
            best_model_state = None
            patience_counter = 0

            for epoch in range(self.detector.epochs):
                avg_train_loss, train_acc = self.detector.classifier.train_step(epoch)

                val_res = self.detector.classifier.val_step(epoch)
                if val_res is None:
                    print()
                    continue

                avg_val_loss, val_acc, precision, recall, f1, f_beta, map_score = (
                    val_res
                )

                if self.detector.selection_metric == "precision":
                    current_metric = precision
                elif self.detector.selection_metric == "f1":
                    current_metric = f1
                elif self.detector.selection_metric == "f_beta":
                    current_metric = f_beta
                elif self.detector.selection_metric == "map":
                    current_metric = map_score
                else:
                    current_metric = avg_val_loss

                improved = False
                if self.detector.selection_metric == "val_loss":
                    improved = current_metric < best_metric
                else:
                    improved = current_metric > best_metric

                if improved:
                    best_metric = current_metric
                    best_model_state = self.detector.classifier.get_state()
                    patience_counter = 0
                    print(
                        f"  → New best {self.detector.selection_metric}: {current_metric:.4f}"
                    )
                else:
                    patience_counter += 1
                    print(f"  → No improvement for {patience_counter} epoch(s)")

                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

                print()

            if best_model_state is not None:
                self.detector.classifier.load_state(best_model_state)
                print(
                    f"Restored best model from round {round_idx} ({self.detector.selection_metric}={best_metric:.4f})"
                )

            is_better_overall = False
            if self.detector.selection_metric == "val_loss":
                is_better_overall = best_metric < best_overall_metric
            else:
                is_better_overall = best_metric > best_overall_metric

            if is_better_overall:
                best_overall_metric = best_metric
                best_overall_state = self.detector.classifier.get_state()
                print(
                    f"★ New overall best! {self.detector.selection_metric}={best_overall_metric:.4f}"
                )

            del X_current, y_current
            gc.collect()

        if best_overall_state is not None:
            self.detector.classifier.load_state(best_overall_state)
            print(f"\n{'=' * 60}")
            print(
                f"Final model: best {self.detector.selection_metric}={best_overall_metric:.4f}"
            )
            print(f"{'=' * 60}")

        self.detector.trained = True
        print("\nBootstrap training complete!")

        if X_val is not None:
            val_acc = self.detector.classifier.evaluate(X_val, y_val)
            print(f"Final validation accuracy: {val_acc:.2f}%")

        del X_pos, y_pos, X_neg_initial, y_neg_initial
        if X_val is not None:
            del X_val, y_val
        gc.collect()

    def normalize_in_place(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape(
            -1, self.detector.feature_resolution, self.detector.feature_resolution, 10
        )
        X = (X - self.detector.mean) / self.detector.std
        return X.reshape(
            -1, self.detector.feature_resolution * self.detector.feature_resolution * 10
        )

    def normalize_chunks(self, X: np.ndarray, chunk_size: int = 5000) -> np.ndarray:
        n_samples = len(X)
        result = np.zeros_like(X, dtype=np.float32)

        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            chunk = X[i:end]
            chunk = chunk.reshape(
                -1,
                self.detector.feature_resolution,
                self.detector.feature_resolution,
                10,
            )
            chunk = (chunk - self.detector.mean) / self.detector.std
            result[i:end] = chunk.reshape(
                -1,
                self.detector.feature_resolution
                * self.detector.feature_resolution
                * 10,
            )

        return result

    def mine_hard_negatives_to_heap(
        self,
        heap_miner: HeapBasedHardNegativeMiner,
        annotation_file: str,
        image_base_dir: str,
        annotation_setting: AnnotationSetting,
        num_images: int = 100,
        max_hard_negs_per_image: int = 50,
        score_threshold: float = 0.0,
    ):
        print(f"\n{'=' * 60}")
        print(f"Mining hard negatives from {num_images} images...")
        print(f"Score threshold: {score_threshold:.4f}")
        print(f"{'=' * 60}")

        annotation_setting = AnnotationSetting(
            None, None, None, None, None, True
        )  # Force it to include all images
        annotations = parse_wider_face_annotation(annotation_file, annotation_setting)
        image_paths = list(annotations.keys())

        mining_images = np.random.choice(
            image_paths, min(num_images, len(image_paths)), replace=False
        )

        total_mined = 0

        for img_path in tqdm(mining_images, desc="Mining hard negatives", ncols=100):
            try:
                image = load_image(img_path, image_base_dir)
                gt_boxes = annotations[img_path]

                hard_negs_this_image = 0

                h, w = image.shape[:2]
                win_w, win_h = self.detector.window_size

                if h >= win_h and w >= win_w:
                    channels = compute_channels(image)

                    stride = max(4, min(win_w, win_h) // 4)
                    windows = generate_sliding_windows((h, w), (win_w, win_h), stride)
                    np.random.shuffle(windows)
                    windows = np.array(windows)

                    iou_matrix = compute_iou_batch(windows, gt_boxes)
                    max_ious = np.max(iou_matrix, axis=1)

                    neg_mask = max_ious <= self.detector.neg_iou_thresh
                    neg_samples = windows[neg_mask]

                    if len(neg_samples) > 0:
                        batch_features = self.detector.extract_features_batch(
                            channels, neg_samples
                        )

                        if len(batch_features) == 0:
                            continue

                        batch_features = np.array(batch_features, dtype=np.float32)
                        features_norm = self.normalize_in_place(batch_features)
                        scores = self.detector.classifier.infer_batch(features_norm)

                        for idx, score in enumerate(scores):
                            if score > score_threshold:
                                heap_miner.add(float(score), features_norm[idx])
                                hard_negs_this_image += 1
                                total_mined += 1

                                if hard_negs_this_image >= max_hard_negs_per_image:
                                    break

            except Exception as e:
                print(f"Error mining from {img_path}: {e}")
                continue

        print(
            f"Mined {total_mined} candidates, heap contains {len(heap_miner)} hardest"
        )
