import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

from acf.channels import compute_channels
from acf.preprocessing import compute_iou_batch


@dataclass
class WIDERFACEConfig:
    annotation_file: str
    image_base_dir: str
    window_size: Tuple[int, int] = (64, 64)
    feature_resolution: int = 8
    pos_iou_thresh: float = 0.5
    neg_iou_thresh: float = 0.3
    hard_neg_iou_range: Tuple[float, float] = (0.1, 0.3)
    num_neg_per_pos: int = 3
    stride_divisor: int = 4
    max_images: Optional[int] = None

    # Attribute filters (None means accept all values)
    acceptable_blur: Optional[List[int]] = None  # 0=clear, 1=normal, 2=heavy
    acceptable_expression: Optional[List[int]] = None  # 0=typical, 1=exaggerate
    acceptable_illumination: Optional[List[int]] = None  # 0=normal, 1=extreme
    acceptable_occlusion: Optional[List[int]] = None  # 0=none, 1=partial, 2=heavy
    acceptable_pose: Optional[List[int]] = None  # 0=typical, 1=atypical
    filter_invalid: bool = True


class WIDERFACEDataset(Dataset):
    def __init__(self, config: WIDERFACEConfig):
        self.config = config
        self.annotations = self._parse_annotations()
        self.image_paths = list(self.annotations.keys())

        if config.max_images:
            self.image_paths = self.image_paths[: config.max_images]

        self.samples = self._build_sample_index()

    def _parse_annotations(self) -> Dict[str, np.ndarray]:
        annotations = {}
        with open(self.config.annotation_file, "r") as f:
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

                values = list(map(int, box_line[:10]))
                if not self._allowed_attribute(values):
                    continue

                boxes.append(values)

            if boxes:
                annotations[img_path] = np.array(boxes, dtype=np.float32)

        return annotations

    def _allowed_attribute(self, values: List[int]) -> bool:
        _, _, w, h = values[:4]
        blur, expression, illumination, invalid, occlusion, pose = values[4:10]

        if w <= 0 or h <= 0:
            return False

        if self.config.filter_invalid and invalid == 1:
            return False

        if (
            self.config.acceptable_blur is not None
            and blur not in self.config.acceptable_blur
        ):
            return False

        if (
            self.config.acceptable_expression is not None
            and expression not in self.config.acceptable_expression
        ):
            return False

        if (
            self.config.acceptable_illumination is not None
            and illumination not in self.config.acceptable_illumination
        ):
            return False

        if (
            self.config.acceptable_occlusion is not None
            and occlusion not in self.config.acceptable_occlusion
        ):
            return False

        if (
            self.config.acceptable_pose is not None
            and pose not in self.config.acceptable_pose
        ):
            return False

        return True

    def _build_sample_index(
        self,
    ) -> List[Tuple[str, Tuple[int, int, int, int], int]]:
        samples = []
        stride = min(self.config.window_size) // self.config.stride_divisor

        print(f"Building sample index for {len(self.image_paths)} images...")

        for img_idx, img_path in enumerate(self.image_paths):
            if img_idx % 100 == 0:
                print(f"Processing image {img_idx}/{len(self.image_paths)}...")

            full_path = os.path.join(self.config.image_base_dir, img_path)
            img = cv2.imread(full_path)
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]
            img_attributes = self.annotations[img_path]
            gt_boxes = img_attributes[:, :4]

            win_w, win_h = self.config.window_size

            if orig_h < win_h or orig_w < win_w:
                continue

            for gt_box in gt_boxes:
                x, y, w, h = gt_box
                if x >= 0 and y >= 0 and x + w <= orig_w and y + h <= orig_h:
                    samples.append((img_path, (int(x), int(y), int(w), int(h)), 1))

            windows = self._generate_windows((orig_h, orig_w), stride)

            if len(windows) == 0:
                continue

            iou_matrix = compute_iou_batch(windows, gt_boxes)
            max_ious = np.max(iou_matrix, axis=1)

            hard_neg_mask = (max_ious >= self.config.hard_neg_iou_range[0]) & (
                max_ious < self.config.hard_neg_iou_range[1]
            )
            neg_mask = max_ious < self.config.neg_iou_thresh

            num_pos = len(gt_boxes)
            num_neg_needed = num_pos * self.config.num_neg_per_pos

            hard_neg_indices = np.where(hard_neg_mask)[0]
            neg_indices = np.where(neg_mask)[0]

            num_hard = min(len(hard_neg_indices), num_neg_needed // 2)
            num_easy = num_neg_needed - num_hard

            if len(hard_neg_indices) > num_hard:
                hard_neg_indices = np.random.choice(
                    hard_neg_indices, num_hard, replace=False
                )

            if len(neg_indices) > num_easy:
                neg_indices = np.random.choice(neg_indices, num_easy, replace=False)

            selected_negs = np.concatenate([hard_neg_indices, neg_indices])
            for idx in selected_negs:
                win = windows[idx]
                samples.append(
                    (
                        img_path,
                        (int(win[0]), int(win[1]), int(win[2]), int(win[3])),
                        0,
                    )
                )

        print(f"Sample index built: {len(samples)} total samples")
        return samples

    def _generate_windows(
        self, image_shape: Tuple[int, int], stride: int
    ) -> np.ndarray:
        height, width = image_shape
        win_w, win_h = self.config.window_size

        y_coords = np.arange(0, height - win_h + 1, stride, dtype=np.float32)
        x_coords = np.arange(0, width - win_w + 1, stride, dtype=np.float32)

        xx, yy = np.meshgrid(x_coords, y_coords)

        num_windows = len(y_coords) * len(x_coords)
        windows = np.empty((num_windows, 4), dtype=np.float32)
        windows[:, 0] = xx.ravel()
        windows[:, 1] = yy.ravel()
        windows[:, 2] = win_w
        windows[:, 3] = win_h

        return windows

    def _extract_features(
        self, channels: np.ndarray, roi: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        channels: [H, W, 10] pre-computed ACF channels
        roi: (x, y, w, h) region of interest
        returns: flattened feature vector
        """
        x, y, w, h = roi
        x, y, w, h = int(x), int(y), int(w), int(h)

        patch = channels[y : y + h, x : x + w]
        resized_to_window = cv2.resize(
            patch, self.config.window_size, interpolation=cv2.INTER_LINEAR
        )
        smoothed = cv2.GaussianBlur(resized_to_window, (3, 3), sigmaX=1)
        resized = cv2.resize(
            smoothed,
            (self.config.feature_resolution, self.config.feature_resolution),
            interpolation=cv2.INTER_AREA,
        )

        return resized.flatten()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        features: [feature_resolution * feature_resolution * 10] flattened features
        label: 0 or 1 (negative or positive)
        """
        img_path, window, label = self.samples[idx]

        full_path = os.path.join(self.config.image_base_dir, img_path)
        image = cv2.imread(full_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        channels = compute_channels(image)
        features = self._extract_features(channels, window)

        return torch.from_numpy(features).float(), torch.tensor(label, dtype=torch.long)
