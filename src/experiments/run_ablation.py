"""Audit every CGMS candidate strategy on each 20-shot support split."""

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.experiments.common import fit_base_models, load_multimodal_data, modality_probabilities, split_mask
from src.models.cgms import CGMSStacking
from src.models.stacking import ProbabilityStacker
from src.utils.io import load_yaml
from src.utils.metrics import auc_score, avg_external


def run(data_path: str, config_path: str, output: str | None = None) -> pd.DataFrame:
    """Report support-selected strategies and query AUC without query selection."""
    config = load_yaml(config_path)
    frame, inputs = load_multimodal_data(data_path)
    models = fit_base_models(frame, inputs)
    val_mask, b_mask, c_mask = (split_mask(frame, name) for name in ("A-val", "B-test", "C-test"))
    stacker = ProbabilityStacker().fit(
        modality_probabilities(models, inputs, val_mask),
        frame.loc[val_mask, "label"].to_numpy(),
    )
    b_probs = modality_probabilities(models, inputs, b_mask)
    c_probs = modality_probabilities(models, inputs, c_mask)
    b_labels = frame.loc[b_mask, "label"].to_numpy(dtype=int)
    c_labels = frame.loc[c_mask, "label"].to_numpy(dtype=int)
    indices = np.arange(c_labels.size)
    rows = []
    for seed in config["seeds"]:
        support, query = train_test_split(
            indices,
            train_size=int(config["support_size"]),
            random_state=int(seed),
            stratify=c_labels,
        )
        bank = CGMSStacking(stacker, list(config["candidate_strategies"]), int(seed)).fit(
            c_probs[support], c_labels[support]
        )
        candidates = {candidate.strategy: candidate for candidate in bank.candidates}
        for strategy, candidate in candidates.items():
            bank.selected = candidate
            b_auc = auc_score(b_labels, bank.predict_proba(b_probs))
            c_auc = auc_score(c_labels[query], bank.predict_proba(c_probs[query]))
            rows.append(
                {
                    "Seed": seed,
                    "Method / Selector": "CGMS candidate bank",
                    "Selected Strategy": strategy,
                    "C-query AUC": c_auc,
                    "Avg External": avg_external(b_auc, c_auc),
                    "Notes": "selected" if strategy == min(candidates.values(), key=lambda x: x.support_score).strategy else "candidate",
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
    parser.add_argument("--output", default="results/toy_ablation_results.csv")
    args = parser.parse_args()
    print(run(args.data, args.config, args.output).to_string(index=False))


if __name__ == "__main__":
    main()
