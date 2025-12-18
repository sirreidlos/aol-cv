import numpy as np
from typing import Any, Dict, Self, Tuple
import lightgbm as lgb
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from tqdm import tqdm

from acf.abstract_model import Model


class LightGBM(Model):
    def __init__(
        self,
        n_estimators=50,
        learning_rate=0.1,
        max_depth=1,
        random_state=None,
        n_jobs=-1,
        num_leaves=2,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    ):
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "subsample_freq": 1,
            "colsample_bytree": colsample_bytree,
            "verbose": verbose,
            "force_col_wise": True,
        }

        self.model = None
        self.n_estimators = n_estimators
        self.epochs = 1  # LightGBM trains once, not in epochs
        self.is_trained = False

    def train_step(self, epoch: int) -> Tuple[float, float]:
        # Only train once - subsequent epochs just return cached results
        if self.is_trained:
            print(f"  Model already trained, using existing model...")
            X_train, y_train = self.dataloader
            preds_proba = self.model.predict(X_train)
            preds = (preds_proba >= 0.5).astype(int)
            acc = 100 * (preds == y_train).mean()
            return 0.0, acc

        X_train, y_train = self.dataloader

        print(f"  Training LightGBM with {self.n_estimators} boosting rounds...")
        print(f"  Using parallel training with {self.params['n_jobs']} threads")

        train_data = lgb.Dataset(X_train, label=y_train)

        # Custom callback for real-time progress updates
        pbar = tqdm(
            total=self.n_estimators, desc="  Boosting iterations", ncols=80, unit="iter"
        )

        def update_progress(env):
            # Update progress bar after each iteration
            pbar.update(env.end_iteration - env.begin_iteration)

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.n_estimators,
            callbacks=[update_progress],
        )

        pbar.close()

        self.is_trained = True

        preds_proba = self.model.predict(X_train)
        preds = (preds_proba >= 0.5).astype(int)

        acc = 100 * (preds == y_train).mean()
        precision = precision_score(y_train, preds, zero_division=0)
        recall = recall_score(y_train, preds, zero_division=0)
        f1 = f1_score(y_train, preds, zero_division=0)
        beta = 0.5
        f_beta = (
            (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-6)
        )
        map_score = average_precision_score(y_train, preds_proba)

        print(f"  ├ Train Accuracy: {acc:.2f}%")
        print(f"  ├ Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"  ├ F1: {f1:.4f}, F-beta: {f_beta:.4f}, mAP: {map_score:.4f}")

        return 0.0, acc

    def val_step(self, epoch: int):
        if self.val_dataloader is None:
            return None

        # If not trained yet, skip validation
        if not self.is_trained:
            print("  Skipping validation (model not trained yet)")
            return None

        X_val, y_val = self.val_dataloader

        print("  Evaluating on validation set...")
        preds_proba = self.model.predict(X_val)
        preds = (preds_proba >= 0.5).astype(int)

        acc = 100 * (preds == y_val).mean()
        precision = precision_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        beta = 0.5
        f_beta = (
            (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-6)
        )
        map_score = average_precision_score(y_val, preds_proba)

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

        # Reset training state
        self.is_trained = False
        self.model = None

    def evaluate(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        if self.model is None:
            return 0.0
        preds_proba = self.model.predict(X_train)
        preds = (preds_proba >= 0.5).astype(int)
        acc = 100 * (preds == y_train).mean()
        return acc

    def infer_batch(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        probs = self.model.predict(features)
        return probs

    def get_state(self) -> Dict[str, Any]:
        if self.model is None:
            return {
                "model_str": None,
                "params": self.params,
                "n_estimators": self.n_estimators,
                "is_trained": self.is_trained,
            }
        model_str = self.model.model_to_string()
        return {
            "model_str": model_str,
            "params": self.params,
            "n_estimators": self.n_estimators,
            "is_trained": self.is_trained,
        }

    def load_state(self, state: Dict[str, Any]) -> Self:
        if state["model_str"] is not None:
            self.model = lgb.Booster(model_str=state["model_str"])
        self.params = state["params"]
        self.n_estimators = state["n_estimators"]
        self.is_trained = state.get("is_trained", False)
        return self
