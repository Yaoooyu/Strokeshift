#!/usr/bin/env bash
set -euo pipefail
python -m src.experiments.run_single_modality --data data/toy_example/manifest.csv "$@"
