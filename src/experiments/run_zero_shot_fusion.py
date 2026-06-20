"""Run zero-shot probability-level multimodal fusion."""

import argparse

import numpy as np
import pandas as pd

from src.experiments.common import fit_base_models, load_multimodal_data, modality_probabilities, split_mask
from src.models.stacking import ProbabilityStacker
from src.utils.metrics import auc_score, avg_external, worst_center_auc


def run(data_path: str, output: str | None = None) -> pd.DataFrame:
    """Fit stacking on A-val and evaluate B/C without target labels."""
    frame, inputs = load_multimodal_data(data_path)
    models = fit_base_models(frame, inputs)
    val_mask = split_mask(frame, "A-val")
    b_mask, c_mask = split_mask(frame, "B-test"), split_mask(frame, "C-test")
    val_probs = modality_probabilities(models, inputs, val_mask)
    stacker = ProbabilityStacker().fit(val_probs, frame.loc[val_mask, "label"].to_numpy())
    methods = {
        "Equal Average": lambda probabilities: probabilities.mean(axis=1),
        "Logistic Stacking": stacker.predict_proba,
        "CT+EHR Stacking": lambda probabilities: ProbabilityStacker().fit(
            val_probs[:, :2], frame.loc[val_mask, "label"].to_numpy()
        ).predict_proba(probabilities[:, :2]),
    }
    b_probs = modality_probabilities(models, inputs, b_mask)
    c_probs = modality_probabilities(models, inputs, c_mask)
    b_labels = frame.loc[b_mask, "label"].to_numpy()
    c_labels = frame.loc[c_mask, "label"].to_numpy()
    rows = []
    for name, predictor in methods.items():
        b_auc = auc_score(b_labels, np.asarray(predictor(b_probs)))
        c_auc = auc_score(c_labels, np.asarray(predictor(c_probs)))
        rows.append(
            {
                "Method": name,
                "B-test AUC": b_auc,
                "C-test AUC": c_auc,
                "Avg External": avg_external(b_auc, c_auc),
                "Worst-center AUC": worst_center_auc(b_auc, c_auc),
                "B-C Gap": b_auc - c_auc,
            }
        )
    result = pd.DataFrame(rows)
    if output:
        result.to_csv(output, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/toy_zero_shot_fusion_results.csv")
    args = parser.parse_args()
    print(run(args.data, args.output).to_string(index=False))


if __name__ == "__main__":
    main()
