"""Probability calibration interfaces."""

import numpy as np
from sklearn.linear_model import LogisticRegression


def probability_to_logit(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert probabilities to numerically stable logits."""
    clipped = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


class PlattCalibrator:
    """Fit Platt/logit calibration on a labeled support set."""

    def __init__(self, random_state: int = 42) -> None:
        self.model = LogisticRegression(max_iter=2000, random_state=random_state)
        self.is_identity = False

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        labels = np.asarray(labels, dtype=int)
        if np.unique(labels).size < 2:
            self.is_identity = True
            return self
        self.model.fit(probability_to_logit(probabilities).reshape(-1, 1), labels)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.is_identity:
            return np.asarray(probabilities, dtype=float)
        logits = probability_to_logit(probabilities).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]

    def fit_transform(self, probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self.fit(probabilities, labels).transform(probabilities)


# TODO: expose additional temperature/isotonic calibrators only after their
# selection protocol is fixed without query-label leakage.
