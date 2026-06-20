# Data Schema

Each aligned sample is represented by one manifest row.

| Field | Type | Description |
|---|---|---|
| `patient_id` | string | De-identified study identifier. Public toy data uses `toy_*` only. |
| `center_id` | categorical | Pseudonymous center code: `A`, `B`, or `C`. |
| `ct_path` | path | Path to a CT volume or an approved CT representation. |
| `ehr_features` | JSON/list or path | Ordered numeric EHR feature vector matching `ehr_variable_list.csv`. |
| `report_text` | string or secure path | Radiology report text available only in the controlled environment. |
| `label` | integer | Binary outcome: `0` non-hemorrhage, `1` hemorrhage. |
| `split` | categorical | `A-train`, `A-val`, `B-test`, or `C-test`. |

All modalities for a row must refer to the same clinical episode. Missingness handling, feature ordering, temporal eligibility, and label definitions must be frozen before external evaluation. Real identifiers and original hospital names must not appear in exported manifests.
