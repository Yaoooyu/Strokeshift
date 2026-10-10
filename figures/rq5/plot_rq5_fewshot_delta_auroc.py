"""Render the RQ5 few-shot AUROC-change figure from frozen summary tables only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.cwd()
TABLES = ROOT / "analysis" / "RQ5" / "tables"
OUT = ROOT / "analysis" / "RQ5" / "figures"
SUMMARY_PATH = TABLES / "rq5_data_efficiency_summary.csv"
BOOTSTRAP_PATH = TABLES / "rq5_paired_bootstrap.csv"

MODELS = [
    "Generic Cross-modal Transformer",
    "DrFuse",
    "MedFuse",
    "M3Care",
    "IRENE",
]
MODEL_LABELS = {
    "Generic Cross-modal Transformer": "Cross-modal Transformer",
    "DrFuse": "DrFuse",
    "MedFuse": "MedFuse",
    "M3Care": "M3Care",
    "IRENE": "IRENE",
}
# Maintains the established RQ4 palette: transformer blue, DrFuse green, M3Care purple.
COLORS = {
    "Generic Cross-modal Transformer": "#4C78A8",
    "DrFuse": "#59A14F",
    "MedFuse": "#D66B62",
    "M3Care": "#9A4D8E",
    "IRENE": "#E28E2C",
}
MARKERS = {
    "Generic Cross-modal Transformer": "o",
    "DrFuse": "s",
    "MedFuse": "D",
    "M3Care": "^",
    "IRENE": "P",
}
EXPECTED_SHOTS = {"B": [5, 10], "C": [5, 10, 20, 40]}


def validate_inputs(summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    required_summary = {"Model", "Center", "Shot", "Delta_AUROC_mean", "Delta_AUROC_std"}
    required_paired = {"Metric", "CI_excludes_zero", "BootstrapValid"}
    if missing := required_summary - set(summary.columns):
        raise ValueError(f"Summary table missing columns: {sorted(missing)}")
    if missing := required_paired - set(paired.columns):
        raise ValueError(f"Paired-bootstrap table missing columns: {sorted(missing)}")
    auroc_pairs = paired.loc[paired["Metric"].eq("AUROC")]
    if len(auroc_pairs) != 210:
        raise ValueError(f"Expected 210 frozen AUROC paired comparisons; found {len(auroc_pairs)}")
    if not auroc_pairs["BootstrapValid"].astype(bool).all():
        raise ValueError("At least one frozen AUROC paired-bootstrap result is invalid")
    if int(auroc_pairs["CI_excludes_zero"].astype(bool).sum()) != 0:
        raise ValueError("Frozen paired-bootstrap table no longer has zero AUROC CIs excluding zero")


def build_plotting_data(summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for center, shots in EXPECTED_SHOTS.items():
        for model in MODELS:
            for shot in shots:
                subset = summary.loc[
                    summary["Model"].eq(model)
                    & summary["Center"].eq(center)
                    & summary["Shot"].eq(shot)
                ]
                if len(subset) != 1:
                    raise ValueError(f"Expected one frozen row for {model}, {center}, {shot}-shot; found {len(subset)}")
                row = subset.iloc[0]
                records.append(
                    {
                        "model": model,
                        "model_display": MODEL_LABELS[model],
                        "center": center,
                        "shot": int(shot),
                        "shot_label": f"{int(shot)}-shot",
                        "delta_auroc": float(row["Delta_AUROC_mean"]),
                        "source_delta_auroc_sd": float(row["Delta_AUROC_std"]),
                        "source_file": SUMMARY_PATH.name,
                    }
                )
    output = pd.DataFrame(records)
    if len(output) != 30 or output.duplicated(["model", "center", "shot"]).any():
        raise ValueError("Plotting data is not a complete unique 5-model B/C grid")

    # Pre-specified report values, checked after rounding only; source precision is retained for plotting.
    expected_rounded = {
        ("B", "DrFuse"): [0.007, 0.009],
        ("B", "Generic Cross-modal Transformer"): [0.007, -0.004],
        ("B", "IRENE"): [-0.007, -0.005],
        ("B", "M3Care"): [-0.002, 0.002],
        ("B", "MedFuse"): [-0.002, -0.002],
        ("C", "DrFuse"): [0.002, 0.001, 0.002, -0.002],
        ("C", "Generic Cross-modal Transformer"): [-0.001, 0.007, 0.009, 0.001],
        ("C", "IRENE"): [-0.011, 0.009, 0.006, 0.010],
        ("C", "M3Care"): [0.003, 0.002, 0.000, 0.002],
        ("C", "MedFuse"): [0.000, -0.001, 0.000, 0.000],
    }
    for (center, model), expected in expected_rounded.items():
        observed = output.loc[(output.center == center) & (output.model == model)].sort_values("shot")["delta_auroc"].tolist()
        if [round(value, 3) for value in observed] != expected:
            raise ValueError(f"Frozen {center}/{model} values differ from specified RQ5 values: {observed}")
    return output


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 8.8,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def draw(plot_data: pd.DataFrame) -> plt.Figure:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.95), sharey=True)
    for ax, center, title in zip(axes, ["B", "C"], ["(a) Center B", "(b) Center C"], strict=True):
        current = plot_data.loc[plot_data.center.eq(center)]
        shots = EXPECTED_SHOTS[center]
        for model in MODELS:
            data = current.loc[current.model.eq(model)].sort_values("shot")
            ax.plot(
                data["shot"],
                data["delta_auroc"],
                color=COLORS[model],
                marker=MARKERS[model],
                markersize=4.7,
                markeredgewidth=0.55,
                markeredgecolor="white",
                label=MODEL_LABELS[model],
                zorder=3,
            )
        ax.axhline(0, color="#333333", linewidth=0.85, zorder=2)
        ax.set_title(title, fontweight="bold", pad=4.0)
        ax.set_xlabel("Target labels (shots)", labelpad=2.5)
        ax.set_xticks(shots)
        ax.set_xticklabels([f"{shot}-shot" for shot in shots])
        ax.set_xlim(min(shots) - 1.5 if center == "B" else 2.5, max(shots) + 1.5 if center == "B" else 42.5)
        ax.set_ylim(-0.0135, 0.0125)
        ax.set_yticks([-0.01, -0.005, 0, 0.005, 0.01])
        ax.grid(axis="y", color="#DFDFDF", linewidth=0.55, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.7, width=0.65, pad=2.0)
    axes[0].set_ylabel(r"$\Delta$AUROC (adapted $-$ zero-shot)", labelpad=2.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=5,
        frameon=False,
        handlelength=1.25,
        handletextpad=0.36,
        columnspacing=0.95,
    )
    fig.subplots_adjust(left=0.095, right=0.995, top=0.87, bottom=0.19, wspace=0.22)
    return fig


def main() -> None:
    if not SUMMARY_PATH.is_file() or not BOOTSTRAP_PATH.is_file():
        raise FileNotFoundError("A required frozen RQ5 table is unavailable")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY_PATH)
    paired = pd.read_csv(BOOTSTRAP_PATH)
    validate_inputs(summary, paired)
    plot_data = build_plotting_data(summary)
    plot_data.to_csv(OUT / "RQ5_fewshot_delta_auroc_plotting_data.csv", index=False, float_format="%.6f")
    figure = draw(plot_data)
    for extension, options in [
        ("pdf", {"bbox_inches": "tight", "pad_inches": 0.03}),
        ("svg", {"bbox_inches": "tight", "pad_inches": 0.03}),
        ("png", {"dpi": 600, "bbox_inches": "tight", "pad_inches": 0.03}),
    ]:
        figure.savefig(OUT / f"RQ5_fewshot_delta_auroc.{extension}", **options)
    plt.close(figure)
    print("Validated 30 frozen plotting means and 210 frozen paired AUROC comparisons (0 CI excludes zero).")
    print(f"Wrote figure and tidy source data to {OUT}")


if __name__ == "__main__":
    main()
