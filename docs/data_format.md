# Data format

Processed NCCT is NIfTI with benchmark IDs as filenames. `ehr/ehr.csv` contains
`benchmark_id` and 24 structured features. `labels/labels.csv` contains
`benchmark_id` and binary `ht_label` (0 = non-HT; 1 = HT).
