"""Probability-level logistic stacking."""

import numpy as np
from sklearn.linear_model import LogisticRegression


class ProbabilityStacker:
    """Combine aligned modality probabilities with logistic regression.

    Args:
        regularization_c: Inverse L2 regularization strength.
        random_state: Reproducibility seed.
    """

    def __init__(self, regularization_c: float = 1.0, random_state: int = 42) -> None:
        self.model = LogisticRegression(
            C=regularization_c,
            max_iter=2000,
            random_state=random_state,
        )

    @staticmethod
    def _validate(probabilities: np.ndarray) -> np.ndarray:
        array = np.asarray(probabilities, dtype=float)
        if array.ndim != 2:
            raise ValueError("Stacking input must have shape (samples, modalities).")
        if not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
            raise ValueError("Stacking inputs must be finite probabilities in [0, 1].")
        return array

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "ProbabilityStacker":
        self.model.fit(self._validate(probabilities), np.asarray(labels, dtype=int))
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._validate(probabilities))[:, 1]
