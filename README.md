# StrokeShift

Official implementation scaffold for **StrokeShift: Center-wise Global Modality Selection for Robust Multimodal Clinical Prediction under Cross-center Domain Shift**.

StrokeShift studies post-thrombolysis hemorrhage prediction across three clinical centers (Hospital A/B/C) using non-contrast head CT, structured EHR variables, and radiology reports. The proposed **CGMS-Stacking** pipeline builds modality-specific predictors, calibrates candidate modality subsets on a small target-center support set, selects a center-wise strategy, and produces probability-level stacked predictions.



## Dataset

The private cohort contains three centers and three aligned modalities:

- **CT:** non-contrast head CT or privacy-preserving CT representations.
- **EHR:** structured clinical variables collected around treatment.
- **Report:** radiology report text or approved text representations.

The repository contains only schema documentation and fully synthetic toy data. See [data/README.md](data/README.md) and [data/data_request.md](data/data_request.md).

## Method

CGMS-Stacking evaluates a bank of calibration strategies (`none`, individual modalities, modality pairs, and all modalities) using 20 labeled support samples from a target center. The selected center-wise strategy is applied to query predictions. Query labels are never used for calibration or strategy selection.

The paper's final instantiation is report-calibrated probability-level logistic stacking. The implementation boundaries are documented in `src/models/cgms.py`, `src/models/calibration.py`, and `src/models/stacking.py`.

## Repository Layout

```text
data/          schema, access instructions, split protocol, synthetic toy data
configs/       reproducible experiment configuration
src/           preprocessing, models, experiments, and utilities
scripts/       shell entry points
results/       paper-table CSV files or explicit placeholders
assets/        publication figures or safe placeholders
checkpoints/   checkpoint access policy; no trained clinical weights
docs/          ethics, access policy, model card, and related work notes
```

## Installation

```bash
conda env create -f environment.yml
conda activate strokeshift
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate the deterministic synthetic dataset:

```bash
python data/toy_example/generate_toy_data.py
```

## Experiments

The commands below default to synthetic data and do not require private clinical data.

```bash
python -m src.experiments.run_single_modality --data data/toy_example/manifest.csv
python -m src.experiments.run_zero_shot_fusion --data data/toy_example/manifest.csv
python -m src.experiments.run_20shot_adaptation --data data/toy_example/manifest.csv --config configs/cgms_20shot.yaml
python -m src.experiments.run_ablation --data data/toy_example/manifest.csv --config configs/cgms_20shot.yaml
```

Equivalent wrappers are available under `scripts/`. Replace the toy manifest only after receiving approval and following the local data governance process.

## Reproducibility Notes

- Formal experiments train on Hospital A and externally evaluate Hospital B/C.
- Zero-shot validation uses no target-center labels.
- The 20-shot setting draws 20 Hospital C support samples and evaluates the remaining query samples for seeds 42-46.
- The toy data validates code paths only; its results have no clinical meaning.

## Dataset

https://pan.quark.cn/s/7d06f9e5575c?pwd=pBBk
Code:pBBk
