"""Structured EHR parsing and imputation."""

import json

import numpy as np
from sklearn.impute import SimpleImputer


def parse_ehr_features(values: list[str]) -> np.ndarray:
    """Parse JSON-encoded feature vectors from a manifest column."""
    rows = [np.asarray(json.loads(value), dtype=float) for value in values]
    if not rows:
        return np.empty((0, 0), dtype=float)
    width = rows[0].shape[0]
    if any(row.ndim != 1 or row.shape[0] != width for row in rows):
        raise ValueError("EHR feature vectors must be aligned and one-dimensional.")
    return np.vstack(rows)


def impute_ehr_features(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Fit median imputation on training data and transform all provided arrays."""
    imputer = SimpleImputer(strategy="median")
    transformed = [imputer.fit_transform(train)]
    transformed.extend(imputer.transform(array) for array in others)
    return tuple(transformed)


# TODO: freeze variable-specific units, ranges, missingness indicators, and
# temporal eligibility rules after institutional review.
