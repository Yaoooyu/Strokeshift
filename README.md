# Synthetic Toy Example

Run `python generate_toy_data.py` to create `manifest.csv` and per-sample synthetic CT representation files under `ct/`. The generator is deterministic and uses no clinical source data.

The toy cohort is intentionally small and statistically artificial. It exercises preprocessing, stacking, support/query splitting, calibration, and metric code; it must not be used to infer clinical performance.
