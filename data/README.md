# Data

This repository does not contain the real StrokeShift CT images, EHR tables, radiology reports, identifiers, or derived patient-level predictions. It provides only:

- a machine-readable schema;
- a fixed split summary;
- an EHR variable-list template;
- controlled-access instructions; and
- a fully synthetic toy dataset for code-path testing.

The synthetic records use `toy_*` identifiers and are not derived from real patients. Run `python data/toy_example/generate_toy_data.py` to regenerate them deterministically.

Approved users should place private data outside the Git repository and pass its manifest path explicitly. Never commit private manifests, local mount paths, free-text reports, DICOM/NIfTI files, or trained clinical checkpoints.
