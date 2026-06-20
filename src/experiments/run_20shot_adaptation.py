"""Run the Hospital C 20-shot support/query adaptation protocol."""

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.experiments.common import fit_base_models, load_multimodal_data, modality_probabilities, split_mask
from src.models.cgms import CGMSStacking
from src.models.stacking import ProbabilityStacker
from src.utils.io import load_yaml
from src.utils.metrics import auc_score, avg_external, binary_metrics, worst_center_auc


def run(data_path: str, config_path: str, output: str | None = None) -> pd.DataFrame:
    """Evaluate CGMS using support labels only and query labels only for metrics."""
    config = load_yaml(config_path)
    frame, inputs = load_multimodal_data(data_path)
    models = fit_base_models(frame, inputs)
    val_mask, b_mask, c_mask = (split_mask(frame, name) for name in ("A-val", "B-test", "C-test"))
    val_probs = modality_probabilities(models, inputs, val_mask)
    stacker = ProbabilityStacker().fit(val_probs, frame.loc[val_mask, "label"].to_numpy())
    b_probs = modality_probabilities(models, inputs, b_mask)
    c_probs = modality_probabilities(models, inputs, c_mask)
    b_labels = frame.loc[b_mask, "label"].to_numpy(dtype=int)
    c_labels = frame.loc[c_mask, "label"].to_numpy(dtype=int)
    support_size = int(config["support_size"])
    if support_size >= c_labels.size:
        raise ValueError("support_size must leave at least one C-query sample.")
    indices = np.arange(c_labels.size)
    rows = []
    for seed in config["seeds"]:
        support, query = train_test_split(
            indices,
            train_size=support_size,
            random_state=int(seed),
            stratify=c_labels,
        )
        cgms = CGMSStacking(stacker, list(config["candidate_strategies"]), int(seed)).fit(
            c_probs[support], c_labels[support]
        )
        b_prediction = cgms.predict_proba(b_probs)
        c_prediction = cgms.predict_proba(c_probs[query])
        b_auc = auc_score(b_labels, b_prediction)
        c_metrics = binary_metrics(c_labels[query], c_prediction)
        rows.append(
            {
                "Seed": seed,
                "Method": config["final_method"],
                "Selected Strategy": cgms.selected.strategy,
                "B-test AUC": b_auc,
                "C-query AUC": c_metrics["AUC"],
                "F1": c_metrics["F1"],
                "Brier": c_metrics["Brier"],
                "ECE": c_metrics["ECE"],
                "Avg External": avg_external(b_auc, c_metrics["AUC"]),
                "Worst-center AUC": worst_center_auc(b_auc, c_metrics["AUC"]),
            }
        )
    result = pd.DataFrame(rows)
    if output:
        result.to_csv(output, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/toy_adaptation_results.csv")
    args = parser.parse_args()
    result = run(args.data, args.config, args.output)
    print(result.to_string(index=False))
    print("\nMean metrics:\n", result.select_dtypes("number").mean().to_string())


if __name__ == "__main__":
    main()
