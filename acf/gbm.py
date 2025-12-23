from typing import Any, Dict, List, Optional, Self, Tuple
import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
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
            print("  Model already trained, using existing model...")
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


class SoftCascadeLightGBM(Model):
    """
    Soft Cascade LightGBM implementation.

    Unlike a hard cascade that rejects examples at each stage, this soft cascade:
    1. Trains multiple LightGBM models (stages) sequentially
    2. Each stage focuses on examples previous stages got wrong or were uncertain about
    3. All stages contribute to the final prediction via weighted averaging
    4. Later stages can have increasing complexity to handle harder examples

    This provides a good balance between model capacity and computational efficiency.
    """

    def __init__(
        self,
        n_stages: int = 3,
        stage_configs: Optional[List[Dict]] = None,
        focus_on_hard_examples: bool = True,
        hard_example_mining: str = "error_uncertainty",
        stage_weights: Optional[List[float]] = None,
        **base_params,
    ):
        """
        Args:
            n_stages: Number of cascade stages (default: 3)
            stage_configs: List of config dicts for each stage. If None, automatically
                          creates configs with increasing complexity
            focus_on_hard_examples: Whether to emphasize hard examples in later stages
            hard_example_mining: Strategy for identifying hard examples:
                - "error": misclassified examples
                - "uncertainty": high uncertainty (near-threshold probabilities)
                - "error_uncertainty": both (default)
            stage_weights: Weights for combining stage predictions. If None, uses
                          exponential weighting favoring later stages
            **base_params: Base LightGBM parameters applied to all stages
        """
        super().__init__()

        self.n_stages = n_stages
        self.focus_on_hard_examples = focus_on_hard_examples
        self.hard_example_mining = hard_example_mining

        # Auto-generate stage configs if not provided
        if stage_configs is None:
            self.stage_configs = []
            for i in range(n_stages):
                config = base_params.copy()
                # Progressive complexity: later stages are more powerful
                config.setdefault("num_leaves", min(2 ** (i + 4), 64))
                config.setdefault("n_estimators", max(30, 150 // (i + 1)))
                config.setdefault("learning_rate", 0.1)
                config.setdefault("min_child_samples", max(10, 30 - i * 5))
                config.setdefault("subsample", min(1.0, 0.75 + i * 0.05))
                self.stage_configs.append(config)
        else:
            assert len(stage_configs) == n_stages, "Must provide config for each stage"
            self.stage_configs = stage_configs

        # Set up stage weights for soft combination
        if stage_weights is None:
            # Exponential weighting: later stages have more influence
            self.stage_weights = np.exp(np.linspace(0, 2, n_stages))
        else:
            self.stage_weights = np.array(stage_weights)
        self.stage_weights /= self.stage_weights.sum()  # Normalize

        # Initialize state
        self.models: List[Optional[lgb.Booster]] = [None] * n_stages
        self.stage_train_scores: List[Dict[str, float]] = []
        self.is_trained = False

    def _compute_sample_weights(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> np.ndarray:
        """
        Compute sample weights for next cascade stage.
        Higher weights for hard/uncertain examples.
        """
        if not self.focus_on_hard_examples:
            return np.ones_like(y_true)

        weights = np.ones(len(y_true))

        # Error component
        if self.hard_example_mining in ["error", "error_uncertainty"]:
            y_pred = (y_proba >= 0.5).astype(int)
            errors = (y_pred != y_true).astype(float)
            weights += errors * 3.0  # Strong emphasis on misclassified

        # Uncertainty component
        if self.hard_example_mining in ["uncertainty", "error_uncertainty"]:
            uncertainty = 1.0 - np.abs(y_proba - 0.5) * 2
            weights += uncertainty * 2.0

        # Exponential scaling and normalization
        weights = np.exp(weights - weights.min())
        weights *= len(weights) / weights.sum()

        return weights

    def _train_stage(
        self,
        stage_idx: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weights: np.ndarray,
    ) -> Dict[str, float]:
        """Train a single cascade stage."""
        config = self.stage_configs[stage_idx]

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "verbose": -1,
            "force_col_wise": True,
        }
        params.update(config)

        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)

        # Progress bar
        pbar = tqdm(
            total=config.get("n_estimators", 100),
            desc=f"    Stage {stage_idx + 1}",
            ncols=80,
            unit="iter",
        )

        def update_progress(env):
            pbar.update(env.end_iteration - env.begin_iteration)

        # Train
        self.models[stage_idx] = lgb.train(
            params,
            train_data,
            num_boost_round=config.get("n_estimators", 100),
            callbacks=[update_progress],
        )

        pbar.close()

        # Evaluate stage
        proba = self.models[stage_idx].predict(X_train)
        preds = (proba >= 0.5).astype(int)

        return {
            "acc": 100 * (preds == y_train).mean(),
            "precision": precision_score(y_train, preds, zero_division=0),
            "recall": recall_score(y_train, preds, zero_division=0),
            "f1": f1_score(y_train, preds, zero_division=0),
        }

    def train_step(self, epoch: int) -> Tuple[float, float]:
        """Train all cascade stages sequentially."""
        if self.is_trained:
            X_train, y_train = self.dataloader
            return 0.0, self._evaluate_stages(X_train, y_train)

        X_train, y_train = self.dataloader

        print(f"\n  Training Soft Cascade LightGBM ({self.n_stages} stages)")
        print(f"  Stage weights: {self.stage_weights.round(3)}")

        sample_weights = np.ones(len(X_train))

        for stage_idx in range(self.n_stages):
            print(f"\n  ┌ Stage {stage_idx + 1}/{self.n_stages}")
            print(
                f"  │ Leaves: {self.stage_configs[stage_idx].get('num_leaves', 'auto')}"
            )
            print(
                f"  │ Estimators: {self.stage_configs[stage_idx].get('n_estimators', 'auto')}"
            )

            # Train stage
            metrics = self._train_stage(stage_idx, X_train, y_train, sample_weights)
            self.stage_train_scores.append(metrics)

            print(f"  │ └ Train: Acc={metrics['acc']:.2f}%, F1={metrics['f1']:.4f}")

            # Prepare next stage weights
            if stage_idx < self.n_stages - 1:
                proba = self.models[stage_idx].predict(X_train)
                sample_weights = self._compute_sample_weights(y_train, proba)

        self.is_trained = True

        # Final evaluation
        acc = self._evaluate_stages(X_train, y_train)
        print(f"\n  └ Cascade train accuracy: {acc:.2f}%")

        return 0.0, acc

    def _evaluate_stages(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate overall cascade accuracy."""
        probs = self.infer_batch(X)
        preds = (probs >= 0.5).astype(int)
        return 100 * (preds == y).mean()

    def val_step(self, epoch: int) -> Optional[Tuple]:
        """Validate cascade on validation set."""
        if self.val_dataloader is None or not self.is_trained:
            return None

        X_val, y_val = self.val_dataloader

        print("\n  Validating Soft Cascade")

        # Individual stage predictions
        stage_probs = []
        for idx, model in enumerate(self.models):
            probs = model.predict(X_val)
            stage_probs.append(probs)

            preds = (probs >= 0.5).astype(int)
            acc = 100 * (preds == y_val).mean()
            print(f"  ├ Stage {idx + 1}: Acc={acc:.2f}%")

        # Combined prediction
        final_probs = np.zeros(len(X_val))
        for probs, weight in zip(stage_probs, self.stage_weights):
            final_probs += probs * weight

        final_preds = (final_probs >= 0.5).astype(int)

        # Metrics
        acc = 100 * (final_preds == y_val).mean()
        precision = precision_score(y_val, final_preds, zero_division=0)
        recall = recall_score(y_val, final_preds, zero_division=0)
        f1 = f1_score(y_val, final_preds, zero_division=0)
        beta = 0.5
        f_beta = (
            (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-6)
        )
        map_score = average_precision_score(y_val, final_probs)

        print(f"  └ Final: Acc={acc:.2f}%, F1={f1:.4f}, mAP={map_score:.4f}")

        return 0.0, acc, precision, recall, f1, f_beta, map_score

    def train_init(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs,
    ):
        """Initialize training data and reset state."""
        self.dataloader = (X_train, y_train)
        if X_val is not None and y_val is not None:
            self.val_dataloader = (X_val, y_val)

        self.is_trained = False
        self.models = [None] * self.n_stages
        self.stage_train_scores = []

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate cascade."""
        if not self.is_trained:
            return 0.0
        return self._evaluate_stages(X, y)

    def infer_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Predict using soft cascade - weighted combination of all stage predictions.
        """
        if not self.is_trained:
            raise ValueError("Cascade not trained yet")

        stage_probs = [model.predict(features) for model in self.models]

        # Soft combination
        final_probs = np.zeros(len(features))
        for probs, weight in zip(stage_probs, self.stage_weights):
            final_probs += probs * weight

        return final_probs

    def get_state(self) -> Dict[str, Any]:
        """Serialize cascade state."""
        state = {
            "n_stages": self.n_stages,
            "focus_on_hard_examples": self.focus_on_hard_examples,
            "hard_example_mining": self.hard_example_mining,
            "stage_configs": self.stage_configs,
            "stage_weights": self.stage_weights.tolist(),
            "is_trained": self.is_trained,
        }

        if self.is_trained:
            state["stage_models"] = [
                model.model_to_string() if model is not None else None
                for model in self.models
            ]
            state["stage_train_scores"] = self.stage_train_scores

        return state

    def load_state(self, state: Dict[str, Any]) -> Self:
        """Deserialize cascade state."""
        # Restore configuration
        self.n_stages = state["n_stages"]
        self.focus_on_hard_examples = state["focus_on_hard_examples"]
        self.hard_example_mining = state["hard_example_mining"]
        self.stage_configs = state["stage_configs"]
        self.stage_weights = np.array(state["stage_weights"])

        # Restore models
        if "stage_models" in state and state["stage_models"][0] is not None:
            self.models = [
                lgb.Booster(model_str=model_str) if model_str else None
                for model_str in state["stage_models"]
            ]
            self.is_trained = state.get("is_trained", False)
            self.stage_train_scores = state.get("stage_train_scores", [])
        else:
            self.models = [None] * self.n_stages
            self.is_trained = False
            self.stage_train_scores = []

        return self
