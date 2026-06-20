"""Evaluation metrics used by StrokeShift experiments."""

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score


def auc_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Return ROC AUC, or NaN when a split contains only one class."""
    labels = np.asarray(labels, dtype=int)
    return float("nan") if np.unique(labels).size < 2 else float(roc_auc_score(labels, probabilities))


def f1_at_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> float:
    """Return binary F1 at a pre-specified probability threshold."""
    predictions = np.asarray(probabilities) >= threshold
    return float(f1_score(labels, predictions, zero_division=0))


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Return mean squared probability error."""
    return float(brier_score_loss(labels, probabilities))


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute equal-width expected calibration error."""
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probabilities, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.any():
            ece += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(ece)


def avg_external(*center_aucs: float) -> float:
    """Return the arithmetic mean across external-center AUCs."""
    return float(np.nanmean(np.asarray(center_aucs, dtype=float)))


def worst_center_auc(*center_aucs: float) -> float:
    """Return the minimum valid external-center AUC."""
    return float(np.nanmin(np.asarray(center_aucs, dtype=float)))


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Return the standard per-center metric bundle."""
    return {
        "AUC": auc_score(labels, probabilities),
        "F1": f1_at_threshold(labels, probabilities),
        "Brier": brier_score(labels, probabilities),
        "ECE": expected_calibration_error(labels, probabilities),
    }
