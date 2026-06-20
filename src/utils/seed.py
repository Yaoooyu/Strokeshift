"""Reproducibility helpers."""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python and NumPy and expose the intended hash seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# TODO: seed framework-specific CUDA backends when the clinical deep-learning
# training code is connected to this public scaffold.
