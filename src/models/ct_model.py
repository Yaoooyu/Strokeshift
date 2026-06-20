"""CT representation classifier interface."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class CTModel:
    """Logistic baseline over fixed CT representations.

    This lightweight model keeps the public example runnable. The clinical
    experiments should inject frozen CT encoder outputs through the same API.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=random_state),
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "CTModel":
        self.model.fit(features, labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)[:, 1]
