"""Safe configuration, manifest, and result I/O."""

import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "patient_id",
    "center_id",
    "ct_path",
    "ehr_features",
    "report_text",
    "label",
    "split",
}


def load_yaml(path: str | Path) -> dict:
    """Load a YAML mapping with safe parsing."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for experiment configuration. Install requirements.txt."
        ) from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load and validate a multimodal manifest without logging row content."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if not set(frame["center_id"]).issubset({"A", "B", "C"}):
        raise ValueError("Public center identifiers must be pseudonymous A/B/C codes.")
    if not set(frame["label"]).issubset({0, 1}):
        raise ValueError("Labels must be binary integers 0/1.")
    return frame


def write_json(data: dict, path: str | Path) -> None:
    """Write an aggregate result dictionary as formatted JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
