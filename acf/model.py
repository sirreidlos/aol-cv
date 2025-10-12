import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from typing import Tuple
import cv2

from .channels import aggregate_channels, compute_channel_pyramid, compute_channels
from .preprocessing import (
    extract_training_samples_sliding,
    parse_wider_face_annotation,
    load_image,
    resize_sample,
    compute_iou,
)


class MLPClassifier(nn.Module):
    def __init__(self, input_size=2560, hidden_sizes=[512, 256], num_classes=2):
        super(MLPClassifier, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ACFDetector:
    def __init__(
        self,
        window_size=(64, 64),
        hidden_sizes=[512, 256],
        feature_resolution=16,
        learning_rate=0.001,
        batch_size=32,
        epochs=10,
    ):
        self.window_size = window_size
        self.hidden_sizes = hidden_sizes
        self.feature_resolution = feature_resolution
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        input_size = 10 * feature_resolution * feature_resolution

        self.classifier = MLPClassifier(
            input_size=input_size, hidden_sizes=hidden_sizes, num_classes=2
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(self.device)

        self.trained = False

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
        aggregated = aggregate_channels(channels, self.feature_resolution)

        return aggregated.flatten()

    def _get_cache_key(
        self,
        num_train_images: int,
        num_val_images: int,
        feature_resolution: int,
        window_size: Tuple[int, int],
    ):
        key_str = f"train_{num_train_images}_val_{num_val_images}_res_{feature_resolution}_win_{window_size[0]}x{window_size[1]}"
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

    def train_step(self, dataloader, epoch, optimizer, criterion):
        self.classifier.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        pbar = tqdm(
            dataloader, desc=f"[TRAIN] Epoch {epoch + 1}/{self.epochs}", ncols=100
        )
        for batch_X, batch_y in pbar:
            outputs = self.classifier(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100 * correct / total:.2f}%",
                }
            )

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

        epoch_acc = 100 * correct / total
        avg_loss = epoch_loss / len(dataloader)
        print(
            f"[TRAIN] Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f}, Accuracy: {epoch_acc:.2f}%",
            end="",
        )

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        pos_acc = (all_preds[all_labels == 1] == 1).mean() * 100
        neg_acc = (all_preds[all_labels == 0] == 0).mean() * 100

        print(f"\n  Pos accuracy: {pos_acc:.2f}%, Neg accuracy: {neg_acc:.2f}%")

        return avg_loss, epoch_acc

    def val_step(self, val_dataloader, criterion, epoch):
        self.classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(
                val_dataloader, desc=f"[VAL] Epoch {epoch + 1}/{self.epochs}", ncols=100
            )
            for batch_X, batch_y in pbar:
                outputs = self.classifier(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{100 * val_correct / val_total:.2f}%",
                    }
                )

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        avg_val_loss = val_loss / len(val_dataloader)
        val_acc = 100 * val_correct / val_total
        print(f" - Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

        pos_acc = (all_preds[all_labels == 1] == 1).mean() * 100
        neg_acc = (all_preds[all_labels == 0] == 0).mean() * 100

        print(f"\n  Pos accuracy: {pos_acc:.2f}%, Neg accuracy: {neg_acc:.2f}%")

        return avg_val_loss, val_acc

    def get_train_data(
        self,
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
        annotations = parse_wider_face_annotation(annotation_file)

        image_paths = list(annotations.keys())
        if max_images:
            image_paths = image_paths[:max_images]

        num_val_images = 0
        if val_annotation_file and val_image_base_dir:
            val_annotations = parse_wider_face_annotation(val_annotation_file)
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
        else:
            # no cache artifacts
            print(f"Cache miss - extracting features from {len(image_paths)} images...")

            X_train = []
            y_train = []
            training_scales = [0.5, 0.75, 1.0, 1.25, 1.5]

            for img_path in tqdm(image_paths, desc="Processing images"):
                try:
                    image = load_image(img_path, image_base_dir)
                    gt_boxes = annotations[img_path]
                    pyramid = compute_channel_pyramid(image, training_scales)

                    pos_count = 0
                    neg_count = 0
                    pos_samples_needed = len(gt_boxes) * 3

                    for scaled_img, _, scale in pyramid:
                        h, w = scaled_img.shape[:2]
                        win_w, win_h = self.window_size

                        if h < win_h or w < win_w:
                            continue

                        pos_samples, neg_samples = extract_training_samples_sliding(
                            scaled_img, gt_boxes, scale=scale
                        )

                        for pos_sample in pos_samples:
                            if pos_count < 5:
                                os.makedirs("debug_pos_samples", exist_ok=True)

                                x, y, w, h = pos_sample
                                patch = scaled_img[y : y + h, x : x + w]

                                orig_x, orig_y = int(x / scale), int(y / scale)
                                orig_w, orig_h = int(w / scale), int(h / scale)
                                orig_win = (orig_x, orig_y, orig_w, orig_h)

                                best_iou = max(
                                    compute_iou(orig_win, gt_box) for gt_box in gt_boxes
                                )

                                cv2.imwrite(
                                    f"debug_pos_samples/pos_{len(X_train)}_iou{best_iou:.2f}.jpg",
                                    cv2.cvtColor(patch, cv2.COLOR_RGB2BGR),
                                )
                            if pos_count >= pos_samples_needed:
                                break

                            features = self.extract_features(scaled_img, pos_sample)
                            X_train.append(features)
                            y_train.append(1)
                            pos_count += 1

                        for neg_sample in neg_samples:
                            if neg_count < 5:
                                os.makedirs("debug_neg_samples", exist_ok=True)

                                x, y, w, h = neg_sample
                                patch = scaled_img[y : y + h, x : x + w]

                                orig_x, orig_y = int(x / scale), int(y / scale)
                                orig_w, orig_h = int(w / scale), int(h / scale)
                                orig_win = (orig_x, orig_y, orig_w, orig_h)

                                best_iou = max(
                                    compute_iou(orig_win, gt_box) for gt_box in gt_boxes
                                )

                                cv2.imwrite(
                                    f"debug_neg_samples/neg_{len(X_train)}_iou{best_iou:.2f}.jpg",
                                    cv2.cvtColor(patch, cv2.COLOR_RGB2BGR),
                                )
                            if neg_count >= pos_count * 3:
                                break

                            features = self.extract_features(scaled_img, neg_sample)
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
            if val_annotation_file and val_image_base_dir:
                print("\nLoading validation data...")
                val_annotations = parse_wider_face_annotation(val_annotation_file)

                print(
                    f"Extracting features from {len(val_image_paths)} validation images..."
                )

                X_val_list = []
                y_val_list = []

                for img_path in tqdm(
                    val_image_paths, desc="Processing validation images"
                ):
                    try:
                        image = load_image(img_path, val_image_base_dir)
                        gt_boxes = val_annotations[img_path]
                        pyramid = compute_channel_pyramid(image, training_scales)

                        pos_count = 0
                        neg_count = 0
                        pos_samples_needed = len(gt_boxes) * 3

                        for scaled_img, _, scale in pyramid:
                            h, w = scaled_img.shape[:2]
                            win_w, win_h = self.window_size

                            if h < win_h or w < win_w:
                                continue

                            pos_samples, neg_samples = extract_training_samples_sliding(
                                scaled_img, gt_boxes, scale=scale
                            )

                            for pos_sample in pos_samples:
                                if pos_count >= pos_samples_needed:
                                    break

                                features = self.extract_features(scaled_img, pos_sample)
                                X_val_list.append(features)
                                y_val_list.append(1)
                                pos_count += 1

                            for neg_sample in neg_samples:
                                if neg_count >= pos_count * 3:
                                    break

                                features = self.extract_features(scaled_img, neg_sample)
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

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_patience=5,
    ):
        print(f"Feature range: [{X_train.min()}, {X_train.max()}]")
        print(f"Feature mean: {X_train.mean()}, std: {X_train.std()}")
        print(f"Any NaN: {np.isnan(X_train).any()}")
        print(f"Any Inf: {np.isinf(X_train).any()}")

        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-8

        self.mean = mean
        self.std = std

        X_train = (X_train - mean) / std
        print(f"Feature range, normalized: [{X_train.min()}, {X_train.max()}]")

        X_tensor = torch.from_numpy(X_train).to(self.device)
        y_tensor = torch.from_numpy(y_train).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_dataloader = None
        if X_val is not None:
            X_val = (X_val - mean) / std
            X_val_tensor = torch.from_numpy(X_val).to(self.device)
            y_val_tensor = torch.from_numpy(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=self.learning_rate)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            avg_train_loss, train_acc = self.train_step(
                dataloader, epoch, optimizer, criterion
            )

            if val_dataloader is None:
                print()
                continue

            avg_val_loss, val_acc = self.val_step(val_dataloader, criterion, epoch)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.classifier.state_dict().copy()
                patience_counter = 0
                print(f"  → New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  → No improvement for {patience_counter} epoch(s)")

                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                    print(f"Restoring best model (val_loss={best_val_loss:.4f})")
                    self.classifier.load_state_dict(best_model_state)
                    break

        self.trained = True
        print("Training complete!")

        self.classifier.eval()
        with torch.no_grad():
            outputs = self.classifier(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            train_acc = 100 * (predicted == y_tensor).sum().item() / len(y_tensor)
            print(f"Final training accuracy: {train_acc:.2f}%")

    def save(self, filepath: str):
        model_data = {
            "model_state_dict": self.classifier.state_dict(),
            "window_size": self.window_size,
            "hidden_sizes": self.hidden_sizes,
            "feature_resolution": self.feature_resolution,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "trained": self.trained,
            "std": self.std,
            "mean": self.mean,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
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

        input_size = 10 * self.feature_resolution * self.feature_resolution
        self.classifier = MLPClassifier(
            input_size=input_size, hidden_sizes=self.hidden_sizes, num_classes=2
        )
        self.classifier.load_state_dict(model_data["model_state_dict"])
        self.classifier.to(self.device)
        self.classifier.eval()

        print(f"Model loaded from {filepath}")
