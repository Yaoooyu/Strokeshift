#!/usr/bin/env bash
set -euo pipefail
python -m src.experiments.run_zero_shot_fusion --data data/toy_example/manifest.csv "$@"
