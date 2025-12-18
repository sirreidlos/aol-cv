"""
PLEASE MOVE THIS TO model.py AFTERWARDS
"""

import numpy as np
from typing import Any, Dict, Self, Tuple
from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def train_step(self, epoch: int) -> Tuple[float, float]:
        pass

    @abstractmethod
    def val_step(
        self, epoch: int
    ) -> Tuple[float, float, float, float, float, float, float] | None:
        pass

    @abstractmethod
    def train_init(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        **kwargs,
    ):
        pass

    @abstractmethod
    def infer_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch of features, emitting a 1D confidence output
        """
        pass

    @abstractmethod
    def evaluate(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> Self:
        pass

    def eval(self):
        pass
