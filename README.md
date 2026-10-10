# StrokeShift

## Overview

StrokeShift is a multi-institutional benchmark for studying the
cross-institution robustness of multimodal clinical data integration.

## Clinical task

Pre-treatment NCCT plus structured EHR are used to predict post-thrombolysis
hemorrhagic transformation (HT).

## Benchmark cohort

| Center | Role | N |
|---|---|---:|
| A | Source | 480 |
| B | External | 41 |
| C | External | 78 |

Center A contains 384 training and 96 validation cases.

## Research questions

RQ1 characterizes cross-center heterogeneity. RQ2 evaluates unimodal transfer.
RQ3 evaluates multimodal integration. RQ4 studies missingness and corruption.
RQ5 studies few-shot target adaptation. RQ6 synthesizes cross-RQ stability.

## Data access

The processed StrokeShift benchmark data are distributed separately from the
source code: `DATA_DOWNLOAD_LINK_TO_BE_ADDED`.

## Expected data structure

```text
StrokeShift_Data_Release/
  processed_ncct/{A,B,C}/
  ehr/ehr.csv
  labels/labels.csv
  splits/
  metadata/
```

## Installation

Create the provided conda environment, set `STROKESHIFT_DATA_ROOT` to the data
release root, then use the scripts and configurations corresponding to the
desired research question. No script in this release requires a server-specific
absolute path.

## Evaluation

AUROC, AUPRC, F1, sensitivity, specificity, Brier score, and ECE are reported.
Centers B and C are evaluation-only and never participate in source model
selection.

## Ethics

Approval No. LYEC2026-0172. See the separately distributed dataset documentation.

## Citation

BibTeX placeholder; final bibliographic metadata will be added before release.
