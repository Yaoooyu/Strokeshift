from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Vector-editable typography and a compact PVLDB-friendly visual style.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def main():
    root = Path(".")
    source = root / "analysis/RQ2/tables/RQ2_3_performance_degradation.csv"
    out_dir = root / "analysis/RQ2/figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read the frozen model-level mean degradation values only; no model outputs are changed.
    degradation = pd.read_csv(source)
    data = degradation.loc[
        degradation["metric"].eq("auroc"),
        ["modality", "model", "external_center", "mean_degradation"],
    ].copy()
    expected_modalities = {"CT", "EHR"}
    expected_centers = {"B", "C"}
    if set(data["modality"]) != expected_modalities or set(data["external_center"]) != expected_centers:
        raise RuntimeError("Unexpected modality or center entries in the frozen RQ2 degradation table.")
    counts = data.groupby(["external_center", "modality"])["model"].nunique()
    if not (counts == 4).all():
        raise RuntimeError(f"Expected four models in every modality/center group; found: {counts.to_dict()}")

    summary = data.groupby(["external_center", "modality"], sort=False)["mean_degradation"].mean()
    values = {
        (center, modality): float(summary.loc[(center, modality)])
        for center in ["B", "C"]
        for modality in ["CT", "EHR"]
    }
    # Frozen-result audit: expected displayed values after rounding to three decimals.
    expected_rounded = {
        ("B", "CT"): 0.082,
        ("B", "EHR"): -0.106,
        ("C", "CT"): 0.223,
        ("C", "EHR"): 0.146,
    }
    observed_rounded = {key: round(value, 3) for key, value in values.items()}
    if observed_rounded != expected_rounded:
        raise RuntimeError(f"Frozen RQ2 audit mismatch: {observed_rounded}")

    centers = ["B", "C"]
    x = np.arange(len(centers), dtype=float)
    width = 0.31
    ct_values = [values[(center, "CT")] for center in centers]
    ehr_values = [values[(center, "EHR")] for center in centers]

    fig, ax = plt.subplots(figsize=(3.55, 2.65), facecolor="white")
    ct_bars = ax.bar(
        x - width / 2,
        ct_values,
        width,
        label="NCCT",
        color="#DAE8FC",
        edgecolor="#91ADD3",
        linewidth=0.65,
        zorder=3,
    )
    ehr_bars = ax.bar(
        x + width / 2,
        ehr_values,
        width,
        label="EHR",
        color="#F8CECC",
        edgecolor="#C88A88",
        linewidth=0.65,
        zorder=3,
    )
    ax.axhline(0, color="#272727", linewidth=1.0, zorder=4)
    ax.yaxis.grid(True, color="#D9D9D9", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x, ["Center B", "Center C"])
    ax.set_ylabel("AUROC degradation (A-val − target)")
    ax.set_ylim(-0.145, 0.275)
    ax.set_yticks([-0.10, 0.00, 0.10, 0.20])
    ax.tick_params(axis="both", labelsize=7.5, length=3, width=0.75)
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=7.5,
        ncol=2,
        handlelength=1.2,
        columnspacing=1.1,
        borderaxespad=0.4,
    )

    for bar in [*ct_bars, *ehr_bars]:
        value = bar.get_height()
        label = f"{value:+.3f}"
        offset = 0.009
        if value >= 0:
            y, va = value + offset, "bottom"
        else:
            y, va = value - offset, "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va=va,
            fontsize=7.2,
            fontweight="bold",
            color="#272727",
        )

    fig.subplots_adjust(left=0.22, right=0.98, top=0.96, bottom=0.20)
    base = out_dir / "RQ2_modality_level_degradation"
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(base.with_suffix(f".{extension}"), dpi=600, facecolor="white")
    plt.close(fig)

    for extension in ["pdf", "svg", "png"]:
        output = base.with_suffix(f".{extension}")
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"Output missing or empty: {output}")
        print(f"{output} ({output.stat().st_size} bytes)")
    print(f"SOURCE: {source}")
    for center in centers:
        for modality in ["CT", "EHR"]:
            print(f"{center} {modality}: {values[(center, modality)]:+.6f}")


if __name__ == "__main__":
    main()
