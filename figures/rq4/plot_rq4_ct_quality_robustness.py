"""Render the RQ4 NCCT resolution-degradation robustness figure from frozen tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.cwd()
RQ4_TABLES = ROOT / "analysis" / "RQ4" / "tables"
RQ3_TABLES = ROOT / "analysis" / "RQ3" / "tables"
OUT = ROOT / "analysis" / "RQ4" / "figures"
CT_DEGRADATION = RQ4_TABLES / "rq4_ct_degradation.csv"
RQ4_CLEAN_MM = RQ4_TABLES / "rq4_modality_missing.csv"
RQ3_SUMMARY = RQ3_TABLES / "rq3_unified_summary.csv"

SEVERITIES = [(0, "Clean"), (1, "Low"), (2, "Medium"), (3, "High")]
CT_MODELS = {
    "ct_SimpleCNN": "SimpleCNN",
    "ct_ResNet18": "ResNet18",
    "ct_DenseNet121": "DenseNet121",
    "ct_SwinTransformer": "Swin Transformer",
}
MM_MODELS = {
    "source_selected_probability_fusion": "Probability fusion",
    "generic": "Cross-modal Transformer",
    "drfuse": "DrFuse",
    "m3care": "M3Care",
}
COLORS_CT = {
    "SimpleCNN": "#E28E2C",
    "ResNet18": "#4C78A8",
    "DenseNet121": "#59A14F",
    "Swin Transformer": "#9A4D8E",
}
COLORS_MM = {
    "Probability fusion": "#E28E2C",
    "Cross-modal Transformer": "#4C78A8",
    "DrFuse": "#59A14F",
    "M3Care": "#9A4D8E",
}
MARKERS = ["o", "s", "D", "^"]


def single(df: pd.DataFrame, source: Path, **filters: object) -> pd.Series:
    mask = np.ones(len(df), dtype=bool)
    for key, value in filters.items():
        mask &= df[key].eq(value)
    result = df.loc[mask]
    if len(result) != 1:
        raise ValueError(f"{source.name}: expected one row for {filters}; found {len(result)}")
    return result.iloc[0]


def frozen_clean_ct(summary: pd.DataFrame, model_key: str) -> pd.Series:
    entity = model_key.removeprefix("ct_")
    return single(summary, RQ3_SUMMARY, group="base_ct", entity=entity, split="A_val", metric="auroc").rename({"n_seeds": "n"})


def frozen_clean_multimodal(clean_mm: pd.DataFrame, summary: pd.DataFrame, model_key: str) -> tuple[pd.Series, Path]:
    if model_key == "source_selected_probability_fusion":
        row = single(
            summary,
            RQ3_SUMMARY,
            group="probability_equal_average",
            entity="CT:DenseNet121 + EHR:RandomForest",
            split="A_val",
            metric="auroc",
        )
        return row.rename({"n_seeds": "n"}), RQ3_SUMMARY
    row = single(
        clean_mm,
        RQ4_CLEAN_MM,
        experiment="modality_missing",
        condition="clean",
        level="whole",
        model=model_key,
        split="A_val",
        metric="auroc",
    )
    return row, RQ4_CLEAN_MM


def build_plotting_data(ct: pd.DataFrame, clean_mm: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for family, models in [("CT-only", CT_MODELS), ("Multimodal", MM_MODELS)]:
        for source_model, display in models.items():
            if family == "CT-only":
                clean = frozen_clean_ct(summary, source_model)
                clean_source = RQ3_SUMMARY
            else:
                clean, clean_source = frozen_clean_multimodal(clean_mm, summary, source_model)
            records.append(
                {
                    "family": family,
                    "model": display,
                    "source_model": source_model,
                    "severity": 0,
                    "severity_label": "Clean",
                    "auroc": float(clean["mean"]),
                    "source_sd": float(clean["sd"]),
                    "n": int(clean["n"]),
                    "source_file": clean_source.name,
                }
            )
            for severity, label in SEVERITIES[1:]:
                row = single(
                    ct,
                    CT_DEGRADATION,
                    experiment="ct_degradation",
                    condition="resolution",
                    level=label,
                    model=source_model,
                    split="A_val",
                    metric="auroc",
                )
                records.append(
                    {
                        "family": family,
                        "model": display,
                        "source_model": source_model,
                        "severity": severity,
                        "severity_label": label,
                        "auroc": float(row["mean"]),
                        "source_sd": float(row["sd"]),
                        "n": int(row["n"]),
                        "source_file": CT_DEGRADATION.name,
                    }
                )
    output = pd.DataFrame(records)
    if len(output) != 32 or output.duplicated(["family", "model", "severity"]).any():
        raise ValueError("Expected a complete unique 2 × 4 × 4 frozen AUROC grid")
    return output


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.1,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 5.9,
            "axes.linewidth": 0.68,
            "lines.linewidth": 1.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_panel(ax: plt.Axes, data: pd.DataFrame, colors: dict[str, str], title: str) -> None:
    for (model, color, marker) in zip(colors, colors.values(), MARKERS, strict=True):
        subset = data.loc[data["model"].eq(model)].sort_values("severity")
        ax.plot(
            subset["severity"],
            subset["auroc"],
            color=color,
            marker=marker,
            markersize=4.6,
            markeredgecolor="white",
            markeredgewidth=0.55,
            label=model,
            zorder=3,
        )
    ax.set_title(title, loc="left", fontweight="bold", pad=5.0)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([label for _, label in SEVERITIES])
    ax.set_xlim(-0.12, 3.12)
    ax.set_ylim(0.45, 0.86)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8])
    ax.grid(axis="y", color="#DFDFDF", linewidth=0.55, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.6, width=0.65, pad=2.0)
    ax.legend(loc="lower left", frameon=False, handlelength=1.35, handletextpad=0.35, labelspacing=0.3, borderaxespad=0.35)


def draw(plot_data: pd.DataFrame) -> plt.Figure:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    plot_panel(
        axes[0],
        plot_data.loc[plot_data["family"].eq("CT-only")],
        COLORS_CT,
        "(a) CT-only Models",
    )
    plot_panel(
        axes[1],
        plot_data.loc[plot_data["family"].eq("Multimodal")],
        COLORS_MM,
        "(b) Multimodal Models",
    )
    axes[0].set_ylabel("A-val AUROC", labelpad=2.5)
    for ax in axes:
        ax.set_xlabel("Resolution degradation severity", labelpad=2.5)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.88, bottom=0.19, wspace=0.22)
    return fig


def main() -> None:
    for path in [CT_DEGRADATION, RQ4_CLEAN_MM, RQ3_SUMMARY]:
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT.mkdir(parents=True, exist_ok=True)
    ct = pd.read_csv(CT_DEGRADATION)
    clean_mm = pd.read_csv(RQ4_CLEAN_MM)
    summary = pd.read_csv(RQ3_SUMMARY)
    data = build_plotting_data(ct, clean_mm, summary)
    data.to_csv(OUT / "RQ4_ct_quality_robustness_plotting_data.csv", index=False, float_format="%.6f")
    fig = draw(data)
    for extension, options in [
        ("pdf", {"bbox_inches": "tight", "pad_inches": 0.03}),
        ("svg", {"bbox_inches": "tight", "pad_inches": 0.03}),
        ("png", {"dpi": 600, "bbox_inches": "tight", "pad_inches": 0.03}),
    ]:
        fig.savefig(OUT / f"RQ4_ct_quality_robustness.{extension}", **options)
    plt.close(fig)
    print("Validated and plotted 32 frozen AUROC values: clean plus Low/Medium/High resolution degradation.")
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
