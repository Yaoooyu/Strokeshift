"""Shared data/model setup for public experiment entry points."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.ct_model import CTModel
from src.models.ehr_model import EHRModel
from src.models.report_model import ReportModel
from src.preprocessing.preprocess_ct import load_ct_features
from src.preprocessing.preprocess_ehr import parse_ehr_features
from src.preprocessing.preprocess_report import normalize_reports
from src.utils.io import load_manifest


MODALITIES = ("CT", "EHR", "Report")


def load_multimodal_data(manifest_path: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a manifest and aligned modality inputs."""
    frame = load_manifest(manifest_path)
    root = Path(manifest_path).resolve().parent
    inputs: dict[str, object] = {
        "CT": load_ct_features(frame["ct_path"].tolist(), root),
        "EHR": parse_ehr_features(frame["ehr_features"].tolist()),
        "Report": normalize_reports(frame["report_text"].tolist()),
    }
    return frame, inputs


def subset_input(value: object, mask: np.ndarray) -> object:
    """Apply a Boolean mask to either array or text-list inputs."""
    if isinstance(value, np.ndarray):
        return value[mask]
    return [item for item, keep in zip(value, mask, strict=True) if keep]


def fit_base_models(
    frame: pd.DataFrame,
    inputs: dict[str, object],
    train_split: str = "A-train",
    seed: int = 42,
) -> dict[str, object]:
    """Fit one public baseline model per modality on Hospital A."""
    mask = frame["split"].to_numpy() == train_split
    labels = frame.loc[mask, "label"].to_numpy(dtype=int)
    models: dict[str, object] = {
        "CT": CTModel(seed),
        "EHR": EHRModel(seed),
        "Report": ReportModel(seed),
    }
    for modality, model in models.items():
        model.fit(subset_input(inputs[modality], mask), labels)
    return models


def modality_probabilities(
    models: dict[str, object],
    inputs: dict[str, object],
    mask: np.ndarray,
) -> np.ndarray:
    """Return aligned CT/EHR/Report probabilities in canonical order."""
    columns = [
        models[modality].predict_proba(subset_input(inputs[modality], mask))
        for modality in MODALITIES
    ]
    return np.column_stack(columns)


def split_mask(frame: pd.DataFrame, split: str) -> np.ndarray:
    """Return a Boolean mask for a named protocol split."""
    return frame["split"].to_numpy() == split
