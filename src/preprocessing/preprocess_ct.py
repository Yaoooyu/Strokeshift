"""CT representation loading.

The public toy pipeline consumes `.npy` feature vectors. Replace this loader
with the approved paper-specific CT preprocessing/encoder inside the secure
environment; never export identifiable images or derived patient paths.
"""

from pathlib import Path

import numpy as np


def load_ct_features(ct_paths: list[str], manifest_dir: str | Path) -> np.ndarray:
    """Load one fixed-length CT representation per manifest row.

    Args:
        ct_paths: Relative or absolute paths to `.npy` feature vectors.
        manifest_dir: Directory used to resolve relative paths.

    Returns:
        Array with shape `(n_samples, representation_dim)`.
    """
    root = Path(manifest_dir)
    vectors = []
    for value in ct_paths:
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        vectors.append(np.asarray(np.load(path), dtype=float).reshape(-1))
    if not vectors:
        return np.empty((0, 0), dtype=float)
    width = vectors[0].shape[0]
    if any(vector.shape[0] != width for vector in vectors):
        raise ValueError("All CT representations must have the same length.")
    return np.vstack(vectors)


# TODO: add the paper's DICOM/NIfTI normalization and frozen CT encoder in a
# private module that is not distributed with patient data or local paths.
