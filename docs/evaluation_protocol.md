# Evaluation protocol

Source training and validation use Center A (384/96). Centers B (41) and C (78)
are external evaluation-only cohorts. The five frozen random seeds are
20260925, 20260926, 20260927, 20260928, and 20260929.

NCCT models: SimpleCNN, ResNet18, DenseNet121, and Swin Transformer.
EHR models: Logistic Regression, Random Forest, XGBoost, and MLP.
Multimodal models: probability fusion, Cross-modal Transformer, MedFuse, M3Care,
DrFuse, and IRENE.

RQ5 uses fixed target support/query splits. Center B has 5-shot and 10-shot
support with a fixed 31-case query set. Center C has 5/10/20/40-shot support
with a fixed 38-case query set. Deep models freeze the source backbone and
adapt the terminal head using AdamW (lr 1e-3, weight decay 1e-4, batch size 2)
for exactly 25 updates. XGBoost and probability fusion use support-only,
rank-preserving Platt calibration. Query labels are evaluation-only.
