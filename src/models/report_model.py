"""Radiology report classifier interface."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


class ReportModel:
    """TF-IDF toy baseline with the same probability interface as MacBERT."""

    def __init__(self, random_state: int = 42) -> None:
        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1),
            LogisticRegression(max_iter=2000, random_state=random_state),
        )

    def fit(self, texts: list[str], labels: np.ndarray) -> "ReportModel":
        self.model.fit(texts, labels)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.model.predict_proba(texts)[:, 1]


# TODO: replace TF-IDF with the frozen/fine-tuned MacBERT representation used
# in the paper while preserving `fit` and `predict_proba` behavior.
