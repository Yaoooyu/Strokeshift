#!/usr/bin/env bash
set -euo pipefail
python -m src.experiments.run_ablation --data data/toy_example/manifest.csv --config configs/cgms_20shot.yaml "$@"
