#!/usr/bin/env bash
set -euo pipefail
python -m src.experiments.run_20shot_adaptation --data data/toy_example/manifest.csv --config configs/cgms_20shot.yaml "$@"
