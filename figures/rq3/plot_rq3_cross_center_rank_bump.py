"""Create the RQ3 cross-center AUROC ranking bump chart from supplied frozen estimates."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path.cwd() / "analysis" / "RQ3" / "figures"
SCENARIOS = ["A-val", "B-test", "C-test", "Worst External"]
MODELS = [
    "EHR XGBoost",
    "Probability fusion",
    "Cross-modal Transformer",
    "DrFuse",
    "MedFuse",
    "M3Care",
    "IRENE",
]

# Frozen RQ3 point estimates provided for the manuscript figure.
AUROC = {
    "EHR XGBoost": [0.744, 0.881, 0.514, 0.514],
    "Probability fusion": [0.842, 0.786, 0.453, 0.453],
    "Cross-modal Transformer": [0.756, 0.567, 0.509, 0.509],
    "DrFuse": [0.765, 0.516, 0.555, 0.516],
    "MedFuse": [0.591, 0.443, 0.498, 0.443],
    "M3Care": [0.813, 0.694, 0.553, 0.553],
    "IRENE": [0.600, 0.507, 0.527, 0.507],
}
EXPECTED_RANKS = {
    "EHR XGBoost": [5, 1, 4, 3],
    "Probability fusion": [1, 2, 7, 6],
    "Cross-modal Transformer": [4, 4, 5, 4],
    "DrFuse": [3, 5, 1, 2],
    "MedFuse": [7, 7, 6, 7],
    "M3Care": [2, 3, 2, 1],
    "IRENE": [6, 6, 3, 5],
}

HIGHLIGHTED = {"Probability fusion", "EHR XGBoost", "M3Care", "DrFuse"}
# Retains the established RQ4 family colours for EHR XGBoost, DrFuse, and M3Care.
COLOURS = {
    "Probability fusion": "#4C78A8",
    "EHR XGBoost": "#E28E2C",
    "M3Care": "#9A4D8E",
    "DrFuse": "#59A14F",
    "Cross-modal Transformer": "#6F8FAE",
    "IRENE": "#9B83B5",
    "MedFuse": "#B88776",
}


def build_data() -> pd.DataFrame:
    wide = pd.DataFrame.from_dict(AUROC, orient="index", columns=SCENARIOS)
    wide.index.name = "model"
    # Worst external is defined as min(B-test, C-test); confirm the frozen input is self-consistent.
    if not np.allclose(wide["Worst External"], wide[["B-test", "C-test"]].min(axis=1)):
        raise ValueError("Worst External does not equal the lower of B-test and C-test AUROC")
    ranks = wide.rank(axis=0, method="min", ascending=False).astype(int)
    for model in MODELS:
        actual = ranks.loc[model, SCENARIOS].tolist()
        if actual != EXPECTED_RANKS[model]:
            raise ValueError(f"Rank assertion failed for {model}: {actual}")
    long = (
        wide.assign(model=wide.index)
        .melt(id_vars="model", value_vars=SCENARIOS, var_name="scenario", value_name="auroc")
        .merge(
            ranks.assign(model=ranks.index).melt(id_vars="model", value_vars=SCENARIOS, var_name="scenario", value_name="rank"),
            on=["model", "scenario"],
            validate="one_to_one",
        )
    )
    long["scenario_order"] = long["scenario"].map({name: idx for idx, name in enumerate(SCENARIOS)})
    long["highlighted"] = long["model"].isin(HIGHLIGHTED)
    return long.sort_values(["model", "scenario_order"]).reset_index(drop=True)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.72,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def draw(data: pd.DataFrame) -> plt.Figure:
    configure_style()
    fig, ax = plt.subplots(figsize=(6.95, 3.45))
    x = np.arange(len(SCENARIOS))

    # Draw muted trajectories first so highlighted models remain dominant.
    for model in [m for m in MODELS if m not in HIGHLIGHTED] + [m for m in MODELS if m in HIGHLIGHTED]:
        part = data.loc[data["model"].eq(model)].sort_values("scenario_order")
        is_highlight = model in HIGHLIGHTED
        ax.plot(
            x,
            part["rank"],
            color=COLOURS[model],
            linewidth=2.25 if is_highlight else 1.15,
            alpha=1.0 if is_highlight else 0.82,
            marker="o",
            markersize=5.8 if is_highlight else 4.3,
            markeredgecolor="white",
            markeredgewidth=0.75,
            zorder=4 if is_highlight else 2,
        )

    ax.set_xlim(-0.15, len(SCENARIOS) - 1 + 0.70)
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS)
    ax.set_ylim(7.45, 0.55)  # Rank 1 at the top.
    ax.set_yticks(range(1, 8))
    ax.set_ylabel("AUROC rank (1 = best)", labelpad=4.0)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.55, zorder=0)
    ax.grid(axis="x", visible=False)
    # Separates center-specific evaluations from the worst-case external summary.
    ax.axvline(2.5, color="#8A8A8A", linewidth=0.8, linestyle=(0, (3.0, 2.3)), zorder=1)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=5.0)
    ax.tick_params(axis="y", length=2.8, width=0.65, pad=2.0)

    # Direct labels at the final scenario prevent the figure from needing a large legend.
    last = data.loc[data["scenario"].eq("Worst External")].sort_values("rank")
    for _, row in last.iterrows():
        model = str(row["model"])
        high = bool(row["highlighted"])
        ax.text(
            3.09,
            float(row["rank"]),
            model,
            va="center",
            ha="left",
            fontsize=7.2 if high else 6.7,
            fontweight="bold" if high else "normal",
            color=COLOURS[model],
            clip_on=False,
            zorder=5,
        )

    fig.subplots_adjust(left=0.095, right=0.80, top=0.96, bottom=0.19)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = build_data()
    data.to_csv(OUT / "RQ3_cross_center_rank_bump_plotting_data.csv", index=False, float_format="%.3f")
    figure = draw(data)
    for suffix, options in [
        ("pdf", {"bbox_inches": "tight", "pad_inches": 0.03}),
        ("svg", {"bbox_inches": "tight", "pad_inches": 0.03}),
        ("png", {"dpi": 600, "bbox_inches": "tight", "pad_inches": 0.03}),
    ]:
        figure.savefig(OUT / f"RQ3_cross_center_rank_bump.{suffix}", **options)
    plt.close(figure)
    print("Rank assertions passed for all 28 model-scenario combinations.")
    print(f"Wrote figure and tidy plotting data to {OUT}")


if __name__ == "__main__":
    main()
