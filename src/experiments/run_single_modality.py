"""Run CT, EHR, and report-only external validation."""

import argparse

import pandas as pd

from src.experiments.common import MODALITIES, fit_base_models, load_multimodal_data, modality_probabilities, split_mask
from src.utils.metrics import auc_score, avg_external, worst_center_auc


def run(data_path: str, output: str | None = None) -> pd.DataFrame:
    """Train on A-train and evaluate each modality on B-test/C-test."""
    frame, inputs = load_multimodal_data(data_path)
    models = fit_base_models(frame, inputs)
    b_mask, c_mask = split_mask(frame, "B-test"), split_mask(frame, "C-test")
    b_probs = modality_probabilities(models, inputs, b_mask)
    c_probs = modality_probabilities(models, inputs, c_mask)
    b_labels = frame.loc[b_mask, "label"].to_numpy()
    c_labels = frame.loc[c_mask, "label"].to_numpy()
    rows = []
    for index, modality in enumerate(MODALITIES):
        b_auc = auc_score(b_labels, b_probs[:, index])
        c_auc = auc_score(c_labels, c_probs[:, index])
        rows.append(
            {
                "Modality": modality,
                "Model": type(models[modality]).__name__,
                "B-test AUC": b_auc,
                "C-test AUC": c_auc,
                "Avg External": avg_external(b_auc, c_auc),
                "Worst-center AUC": worst_center_auc(b_auc, c_auc),
            }
        )
    result = pd.DataFrame(rows)
    if output:
        result.to_csv(output, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the aligned manifest CSV.")
    parser.add_argument("--output", default="results/toy_single_modality_results.csv")
    args = parser.parse_args()
    print(run(args.data, args.output).to_string(index=False))


if __name__ == "__main__":
    main()
