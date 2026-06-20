# Model Card: CGMS-Stacking

## Intended Use

Research on robust multimodal prediction under cross-center domain shift. The evaluated endpoint is post-thrombolysis hemorrhage prediction.

## Inputs and Outputs

Inputs are aligned CT representations, structured EHR features, and radiology-report representations. Output is a probability for a binary hemorrhage outcome.

## Training and Evaluation

Models are trained at center A and externally evaluated at centers B/C. In the 20-shot setting, 20 labeled target-center support samples may be used for calibration or strategy selection; the remaining query labels are evaluation-only.

## Limitations

- Retrospective multi-center results may not generalize to new populations or workflows.
- Calibration and subgroup performance may change under prevalence, acquisition, documentation, and treatment shifts.
- Missing modalities and data-quality failures require explicit handling.
- The toy implementation is not the clinical model and has no clinical validity.

## Prohibited Use

Do not use this code or any resulting model for clinical care, autonomous decisions, patient re-identification, or use outside approved governance.

## Release Checklist

Add final cohort details, subgroup analyses, confidence intervals, failure modes, model version, approval scope, and contact information before release.
