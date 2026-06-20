"""Generate a deterministic, fully synthetic multimodal StrokeShift cohort."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent


def generate(seed: int = 2026) -> Path:
    """Create a toy manifest and synthetic CT vectors; return manifest path."""
    rng = np.random.default_rng(seed)
    ct_dir = OUTPUT_DIR / "ct"
    ct_dir.mkdir(parents=True, exist_ok=True)
    protocol = [
        ("A-train", "A", 48, 24),
        ("A-val", "A", 24, 12),
        ("B-test", "B", 24, 8),
        ("C-test", "C", 40, 20),
    ]
    center_shift = {"A": 0.0, "B": 0.7, "C": -0.6}
    report_templates = {
        0: ["no acute hemorrhage stable appearance", "no intracranial bleeding on synthetic scan"],
        1: ["synthetic hemorrhage finding present", "acute bleeding pattern in toy example"],
    }
    rows = []
    sample_number = 0
    for split, center, size, positives in protocol:
        labels = np.array([1] * positives + [0] * (size - positives), dtype=int)
        rng.shuffle(labels)
        for label in labels:
            sample_number += 1
            patient_id = f"toy_{sample_number:04d}"
            signal = 1.25 * label - 0.4
            ct_vector = rng.normal(center_shift[center] + signal, 1.0, size=8)
            ehr_vector = rng.normal(0.5 * center_shift[center] + 0.9 * label, 1.1, size=6)
            ct_name = f"ct/{patient_id}.npy"
            np.save(OUTPUT_DIR / ct_name, ct_vector.astype(np.float32))
            report = rng.choice(report_templates[int(label)])
            report += f" center_{center.lower()} synthetic documentation"
            rows.append(
                {
                    "patient_id": patient_id,
                    "center_id": center,
                    "ct_path": ct_name,
                    "ehr_features": json.dumps(ehr_vector.round(5).tolist()),
                    "report_text": report,
                    "label": int(label),
                    "split": split,
                }
            )
    manifest = OUTPUT_DIR / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


if __name__ == "__main__":
    generate()
    print("data/toy_example/manifest.csv")
