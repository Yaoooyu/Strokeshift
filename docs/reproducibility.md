# Reproducibility

The `scripts/` directory preserves the scripts used for RQ1--RQ6 and the
`configs/` directory preserves experiment recipes. Aggregate result tables and
publication figures are included when they do not carry patient-level fields.
Patient-level predictions, checkpoints, logs, native CT volumes, and EHR/label
records are intentionally absent from this code package.

Third-party official implementations for DrFuse, MedFuse, M3Care, and IRENE are
not redistributed here. Their documented project references and the StrokeShift
adaptation code are retained in the RQ3-SOTA scripts/configuration.
