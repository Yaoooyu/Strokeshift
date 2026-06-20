"""Structured EHR classifier interface."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class EHRModel:
    """Median-imputed logistic baseline for aligned EHR vectors."""

    def __init__(self, random_state: int = 42) -> None:
        self.model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=random_state),
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "EHRModel":
        self.model.fit(features, labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)[:, 1]
