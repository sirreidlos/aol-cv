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
from typing import List, Optional

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


class AdaBoostCascade(Model):
    def __init__(
        self,
        n_estimators=50,
        learning_rate=1.0,
        max_depth=1,
        random_state=None,
        cascade_thresholds: Optional[List[float]] = None,
        cascade_stages: Optional[List[int]] = None,
    ):
        """
        Wrap sklearn AdaBoostClassifier to conform with the Model interface.

        Args:
            cascade_thresholds: Confidence thresholds for early exit at each stage.
                               If None, no cascading is used.
            cascade_stages: Number of estimators to use at each stage.
                           If None, defaults to evenly spaced stages.
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

        # Cascade configuration
        self.cascade_thresholds = cascade_thresholds
        self.cascade_stages = cascade_stages

        # Statistics for cascade performance
        self.cascade_stats = {"stage_exits": [], "avg_estimators_used": 0.0}

    def _setup_cascade_stages(self):
        """Setup cascade stages if not explicitly provided."""
        if self.cascade_thresholds is None:
            return

        n_estimators = len(self.model.estimators_)

        if self.cascade_stages is None:
            # Default: evenly spaced stages
            n_stages = len(self.cascade_thresholds) + 1
            self.cascade_stages = [
                int(n_estimators * (i + 1) / n_stages) for i in range(n_stages - 1)
            ]
            self.cascade_stages.append(n_estimators)

        # Initialize statistics
        self.cascade_stats["stage_exits"] = [0] * len(self.cascade_stages)

    def _staged_predict_proba(self, X: np.ndarray, n_estimators: int) -> np.ndarray:
        """
        Predict probabilities using only the first n_estimators.

        This is a custom implementation since sklearn's staged_predict_proba
        iterates through all stages.
        """
        if n_estimators == 0:
            # Return prior probabilities
            n_samples = X.shape[0]
            n_classes = len(self.model.classes_)
            return np.full((n_samples, n_classes), 1.0 / n_classes)

        # Get predictions from first n_estimators
        estimators = self.model.estimators_[:n_estimators]
        estimator_weights = self.model.estimator_weights_[:n_estimators]

        n_samples = X.shape[0]
        n_classes = len(self.model.classes_)

        # Initialize predictions
        pred = np.zeros((n_samples, n_classes))

        # Aggregate predictions
        for estimator, weight in zip(estimators, estimator_weights):
            current_pred = estimator.predict(X)
            # Convert to class indices
            class_indices = np.searchsorted(self.model.classes_, current_pred)
            # Add weighted vote
            for i, class_idx in enumerate(class_indices):
                pred[i, class_idx] += weight

        # Normalize to get probabilities
        pred /= pred.sum(axis=1)[:, np.newaxis]

        return pred

    def infer_batch_cascade(
        self, features: np.ndarray, return_stats: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Perform cascaded inference with early exit based on confidence.

        Args:
            features: Input features
            return_stats: If True, return statistics about cascade usage

        Returns:
            probs: Predicted probabilities for positive class
            stats: Dictionary with cascade statistics (if return_stats=True)
        """
        if self.cascade_thresholds is None:
            # No cascade, use standard inference
            probs = self.model.predict_proba(features)[:, 1]
            return (probs, None) if return_stats else probs

        n_samples = features.shape[0]
        probs = np.zeros(n_samples)
        decided_mask = np.zeros(n_samples, dtype=bool)
        estimators_used = np.zeros(n_samples, dtype=int)

        # Process each cascade stage
        for stage_idx, (n_estimators, threshold) in enumerate(
            zip(self.cascade_stages, self.cascade_thresholds)
        ):
            if decided_mask.all():
                break

            # Get undecided samples
            undecided_mask = ~decided_mask
            X_undecided = features[undecided_mask]

            # Predict using this stage's estimators
            stage_probs = self._staged_predict_proba(X_undecided, n_estimators)
            stage_probs_pos = stage_probs[:, 1]

            # Compute confidence (distance from 0.5)
            confidence = np.abs(stage_probs_pos - 0.5)

            # Determine which samples can exit at this stage
            confident_mask = confidence >= threshold

            # Update results for confident samples
            undecided_indices = np.where(undecided_mask)[0]
            confident_indices = undecided_indices[confident_mask]

            probs[confident_indices] = stage_probs_pos[confident_mask]
            decided_mask[confident_indices] = True
            estimators_used[confident_indices] = n_estimators

            # Update statistics
            self.cascade_stats["stage_exits"][stage_idx] += confident_mask.sum()

        # Handle remaining undecided samples with full model
        if not decided_mask.all():
            undecided_mask = ~decided_mask
            X_undecided = features[undecided_mask]

            full_probs = self.model.predict_proba(X_undecided)[:, 1]
            probs[undecided_mask] = full_probs
            estimators_used[undecided_mask] = len(self.model.estimators_)

            # Update statistics for final stage
            self.cascade_stats["stage_exits"][-1] += undecided_mask.sum()

        # Update average estimators used
        self.cascade_stats["avg_estimators_used"] = estimators_used.mean()

        if return_stats:
            stats = {
                "estimators_used": estimators_used,
                "avg_estimators_used": estimators_used.mean(),
                "stage_distribution": self.cascade_stats["stage_exits"].copy(),
                "speedup": len(self.model.estimators_) / estimators_used.mean(),
            }
            return probs, stats

        return probs

    def train_step(self, epoch: int) -> Tuple[float, float]:
        X_train, y_train = self.dataloader

        self.model.fit(X_train, y_train)

        # Setup cascade stages after training
        self._setup_cascade_stages()

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

        # Report cascade configuration
        if self.cascade_thresholds is not None:
            print(f"  ├ Cascade stages: {self.cascade_stages}")
            print(f"  └ Cascade thresholds: {self.cascade_thresholds}")

        return 0.0, acc

    def val_step(self, epoch: int):
        if self.val_dataloader is None:
            return None

        X_val, y_val = self.val_dataloader

        # Use cascade inference if available
        if self.cascade_thresholds is not None:
            probs, stats = self.infer_batch_cascade(X_val, return_stats=True)
            preds = (probs >= 0.5).astype(int)

            print(
                f"  ├ Cascade avg estimators: {stats['avg_estimators_used']:.1f}/{len(self.model.estimators_)}"
            )
            print(f"  ├ Speedup: {stats['speedup']:.2f}x")
        else:
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
        """
        Standard inference without cascade.
        For cascade inference, use infer_batch_cascade() instead.
        """
        probs = self.model.predict_proba(features)[:, 1]
        return probs

    def get_state(self) -> Dict[str, Any]:
        state = self.model.__dict__.copy()
        state["cascade_thresholds"] = self.cascade_thresholds
        state["cascade_stages"] = self.cascade_stages
        return state

    def load_state(self, state: Dict[str, Any]) -> Self:
        self.cascade_thresholds = state.pop("cascade_thresholds", None)
        self.cascade_stages = state.pop("cascade_stages", None)
        self.model.__dict__.update(state)
        self._setup_cascade_stages()
        return self
