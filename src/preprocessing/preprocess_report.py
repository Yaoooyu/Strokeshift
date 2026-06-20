"""Radiology report preprocessing for the public toy pipeline."""

import re


def normalize_report(text: str) -> str:
    """Apply conservative whitespace normalization to synthetic report text."""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_reports(texts: list[str]) -> list[str]:
    """Normalize a list of reports without logging their content."""
    return [normalize_report(text) for text in texts]


# TODO: integrate the paper's locally hosted MacBERT tokenizer/encoder. Do not
# send protected report text to external services.
