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
from sklearn.metrics import average_precision_score

from acf.abstract_model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_flush_denormal(True)


class MLPClassifier(nn.Module, Model):
    def __init__(self, input_size, hidden_sizes, num_classes=2):
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

    def train_step(self, epoch):
        dataloader = self.dataloader
        optimizer = self.optimizer
        criterion = self.criterion

        self.train()
        epoch_loss = 0.0

        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(dataloader, desc=f"T ┬ Epoch {epoch + 1}/{self.epochs}", ncols=100)
        for batch_X, batch_y in pbar:
            outputs = self(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].detach().cpu().numpy())

            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        avg_loss = epoch_loss / len(dataloader)
        epoch_acc = 100 * (all_preds == all_labels).mean()

        true_positives = ((all_preds == 1) & (all_labels == 1)).sum()
        false_positives = ((all_preds == 1) & (all_labels == 0)).sum()
        false_negatives = ((all_preds == 0) & (all_labels == 1)).sum()

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        beta = 0.5
        f_beta = (
            (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-6)
        )

        pos_acc = (all_preds[all_labels == 1] == 1).mean() * 100
        neg_acc = (all_preds[all_labels == 0] == 0).mean() * 100

        print(f"  ├ Train Loss: {avg_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%")
        print(f"  ├ Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"  ├ F1: {f1:.4f}, F-beta: {f_beta:.4f}")
        print(f"  └ Pos accuracy: {pos_acc:.2f}%, Neg accuracy: {neg_acc:.2f}%")

        return avg_loss, epoch_acc

    def val_step(self, epoch):
        val_dataloader = self.val_dataloader

        if val_dataloader is None:
            return None

        criterion = self.criterion
        self.eval()
        val_loss = 0.0

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            pbar = tqdm(
                val_dataloader,
                desc=f"V ┬ Epoch {epoch + 1}/{self.epochs}",
                ncols=100,
            )
            for batch_X, batch_y in pbar:
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())

                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        avg_val_loss = val_loss / len(val_dataloader)
        val_acc = 100 * (all_preds == all_labels).mean()

        true_positives = ((all_preds == 1) & (all_labels == 1)).sum()
        false_positives = ((all_preds == 1) & (all_labels == 0)).sum()
        false_negatives = ((all_preds == 0) & (all_labels == 1)).sum()

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        beta = 0.5
        f_beta = (
            (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-6)
        )

        all_labels = np.array(all_labels, dtype=int)
        all_probs = np.array(all_probs, dtype=float)

        # kinda misleading, but it should be correct to consider this as map
        # because it only has one class to classify
        map_score = average_precision_score(all_labels, all_probs)

        pos_acc = (all_preds[all_labels == 1] == 1).mean() * 100
        neg_acc = (all_preds[all_labels == 0] == 0).mean() * 100

        print(f"  ├ Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        print(f"  ├ Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"  ├ F1: {f1:.4f}, F-beta: {f_beta:.4f} mAP: {map_score:.4f}")
        print(f"  └ Pos accuracy: {pos_acc:.2f}%, Neg accuracy: {neg_acc:.2f}%")

        return avg_val_loss, val_acc, precision, recall, f1, f_beta, map_score

    def train_init(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size=32,
        learning_rate=0.0001,
        epochs=10,
    ):
        self.to(DEVICE)
        self.epochs = epochs

        X_tensor = torch.from_numpy(X_train).to(DEVICE)
        y_tensor = torch.from_numpy(y_train).to(DEVICE)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        val_dataloader = None
        if X_val is not None:
            X_val_tensor = torch.from_numpy(X_val).to(DEVICE)
            y_val_tensor = torch.from_numpy(y_val).to(DEVICE)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )

        # pos_weight = len(y_train) / (2 * np.sum(y_train == 1))
        # neg_weight = len(y_train) / (2 * np.sum(y_train == 0))
        pos_weight = 1
        neg_weight = 1

        class_weights = torch.tensor([neg_weight, pos_weight], dtype=torch.float32).to(
            DEVICE
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer

    def evaluate(self, X_train, y_train):
        X_tensor = torch.from_numpy(X_train).to(DEVICE)
        y_tensor = torch.from_numpy(y_train).to(DEVICE)

        self.eval()
        with torch.no_grad():
            outputs = self(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            train_acc = 100 * (predicted == y_tensor).sum().item() / len(y_tensor)

        return train_acc

    def infer_batch(self, features: np.ndarray) -> np.ndarray:
        # torch.set_flush_denormal(True)
        self.eval()
        with torch.no_grad():
            features_tensor = torch.from_numpy(features).float().to(DEVICE)

            outputs = self(features_tensor)

            probs = torch.softmax(outputs, dim=1)
            scores = probs[:, 1].cpu().numpy()

        return scores

    def get_state(self):
        return self.state_dict().copy()

    def load_state(self, state):
        return self.load_state_dict(state, strict=False)
