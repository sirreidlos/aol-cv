import numpy as np
from typing import Any, Dict, Self, Tuple
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

from acf.abstract_model import Model


class AdaBoost(Model):
    def __init__(
        self, n_estimators=50, learning_rate=1.0, max_depth=1, random_state=None
    ):
        """
        Wrap sklearn AdaBoostClassifier to conform with the Model interface.
        """
        base_estimator = DecisionTreeClassifier(
            max_depth=max_depth, random_state=random_state
        )
        self.model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.epochs = 1  # sklearn AdaBoost is not epoch-based

    def train_step(self, epoch: int) -> Tuple[float, float]:
        X_train, y_train = self.dataloader

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_train)
        probs = self.model.predict_proba(X_train)[:, 1]

        acc = 100 * (preds == y_train).mean()
        precision = precision_score(y_train, preds, zero_division=0)
        recall = recall_score(y_train, preds, zero_division=0)
        f1 = f1_score(y_train, preds, zero_division=0)
        beta = 0.5
        f_beta = (
            (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-6)
        )
        map_score = average_precision_score(y_train, probs)

        print(f"  ├ Train Accuracy: {acc:.2f}%")
        print(f"  ├ Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"  ├ F1: {f1:.4f}, F-beta: {f_beta:.4f}, mAP: {map_score:.4f}")

        return 0.0, acc

    def val_step(self, epoch: int):
        if self.val_dataloader is None:
            return None

        X_val, y_val = self.val_dataloader
        preds = self.model.predict(X_val)
        probs = self.model.predict_proba(X_val)[:, 1]

        acc = 100 * (preds == y_val).mean()
        precision = precision_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        beta = 0.5
        f_beta = (
            (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-6)
        )
        map_score = average_precision_score(y_val, probs)

        print(f"  ├ Val Accuracy: {acc:.2f}%")
        print(f"  ├ Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"  ├ F1: {f1:.4f}, F-beta: {f_beta:.4f}, mAP: {map_score:.4f}")

        return 0.0, acc, precision, recall, f1, f_beta, map_score

    def train_init(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs,
    ):
        self.dataloader = (X_train, y_train)
        if X_val is not None and y_val is not None:
            self.val_dataloader = (X_val, y_val)

    def evaluate(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        preds = self.model.predict(X_train)
        acc = 100 * (preds == y_train).mean()
        return acc

    def infer_batch(self, features: np.ndarray) -> np.ndarray:
        probs = self.model.predict_proba(features)[:, 1]
        return probs

    def get_state(self) -> Dict[str, Any]:
        return self.model.__dict__.copy()

    def load_state(self, state: Dict[str, Any]) -> Self:
        self.model.__dict__.update(state)
        return self
