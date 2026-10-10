from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
    }
)


LABELS = {
    "age": "Age",
    "gender": "Gender",
    "admission_nihss": "Admission NIHSS",
    "admission_sbp": "Admission SBP",
    "admission_dbp": "Admission DBP",
    "admission_glucose": "Admission glucose",
    "dnt_time": "DNT",
    "htn_history": "Hypertension history",
    "hypertension_history": "Hypertension history",
    "dyslipidemia_history": "Dyslipidemia history",
    "smoking_history": "Smoking history",
    "alcohol_history": "Alcohol history",
    "prior_stroke_history": "Prior stroke history",
    "arterial_plaque": "Arterial plaque",
    "hypertension_drug": "Antihypertensive drug",
    "stent_implantation": "Stent implantation",
    "hd_MI": "Myocardial infarction",
    "hd_Other_HD": "Other heart disease",
    "spacing_z": "Z-axis spacing",
    "num_slices": "Number of slices",
    "voxel_count": "Voxel count",
    "inplane_resolution_mm": "In-plane resolution",
    "axial_coverage_mm": "Axial coverage",
    "slice_thickness": "Slice thickness",
}


def label_of(variable):
    return LABELS.get(variable, variable.replace("_", " "))


def cmap_with_na(name):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("#E6E6E6")
    return cmap


def annotation_color(cmap, norm, value):
    rgba = cmap(norm(value))
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if luminance < 0.52 else "#202020"


def heatmap(ax, values, rows, columns, cmap, norm, fmt, cell_fontsize=6.5):
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, fontsize=7)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.15)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isnan(value):
                ax.text(
                    column,
                    row,
                    "NA",
                    ha="center",
                    va="center",
                    fontsize=cell_fontsize,
                    color="#5E5E5E",
                    fontstyle="italic",
                )
            else:
                ax.text(
                    column,
                    row,
                    fmt.format(value),
                    ha="center",
                    va="center",
                    fontsize=cell_fontsize,
                    color=annotation_color(cmap, norm, value),
                )
    return image


def save(fig, destination):
    fig.savefig(destination, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    root = Path(".")
    out_dir = root / "analysis/RQ1/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    ehr_shift = pd.read_csv(root / "analysis/RQ1/tables/figure_RQ1a_ehr_shift_corrected.csv")
    ehr_missing = pd.read_csv(root / "analysis/RQ1/tables/figure_RQ1b_ehr_missingness_corrected.csv")
    ct_shift = pd.read_csv(root / "analysis/RQ1/tables/figure_RQ1c_ct_shift.csv")
    pairs = ["AB", "AC", "BC"]
    pair_labels = ["A–B", "A–C", "B–C"]

    # Panel A — no panel title; concise internal labels preserve the two metric definitions.
    continuous_vars = [
        "age",
        "admission_nihss",
        "admission_glucose",
        "dnt_time",
        "admission_sbp",
        "admission_dbp",
    ]
    categorical_vars = [
        "hypertension_drug",
        "dyslipidemia_history",
        "arterial_plaque",
        "hd_Other_HD",
        "hd_MI",
        "htn_history",
        "gender",
        "smoking_history",
        "alcohol_history",
        "prior_stroke_history",
    ]
    shift_index = ehr_shift.set_index("variable")
    continuous_vars = [v for v in continuous_vars if v in shift_index.index]
    categorical_vars = [v for v in categorical_vars if v in shift_index.index]
    continuous = np.array(
        [np.abs(pd.to_numeric(shift_index.loc[continuous_vars, f"{pair}_SMD"], errors="coerce")) for pair in pairs]
    ).T
    categorical = np.array(
        [pd.to_numeric(shift_index.loc[categorical_vars, f"{pair}_effect_size"], errors="coerce") for pair in pairs]
    ).T
    cmap_smd, cmap_v = cmap_with_na("Blues"), cmap_with_na("Purples")
    norm_smd = Normalize(0, max(0.60, float(np.nanmax(continuous))))
    norm_v = Normalize(0, max(0.25, float(np.nanmax(categorical))))
    fig_a = plt.figure(figsize=(3.5, 5.2), facecolor="white")
    grid = fig_a.add_gridspec(2, 1, height_ratios=[len(continuous_vars), len(categorical_vars)], hspace=0.52)
    ax_a1, ax_a2 = fig_a.add_subplot(grid[0]), fig_a.add_subplot(grid[1])
    image_smd = heatmap(
        ax_a1, continuous, [label_of(v) for v in continuous_vars], pair_labels, cmap_smd, norm_smd, "{:.2f}", 6.2
    )
    image_v = heatmap(
        ax_a2, categorical, [label_of(v) for v in categorical_vars], pair_labels, cmap_v, norm_v, "{:.2f}", 6.1
    )
    cbar_smd = fig_a.colorbar(image_smd, ax=ax_a1, orientation="horizontal", fraction=0.085, pad=0.19)
    cbar_smd.set_label("Absolute pairwise SMD", fontsize=6.7, labelpad=2)
    cbar_smd.ax.tick_params(labelsize=6, length=2, pad=1)
    cbar_v = fig_a.colorbar(image_v, ax=ax_a2, orientation="horizontal", fraction=0.065, pad=0.16)
    cbar_v.set_label("Cramér's V", fontsize=6.7, labelpad=2)
    cbar_v.ax.tick_params(labelsize=6, length=2, pad=1)
    save(fig_a, out_dir / "RQ1_cross_center_heterogeneity_panel_a.png")

    # Panel B — strictly the eight requested EHR variables, in the requested order.
    missing_vars = [
        "prior_stroke_history",
        "admission_glucose",
        "dnt_time",
        "admission_nihss",
        "alcohol_history",
        "arterial_plaque",
        "smoking_history",
        "stent_implantation",
    ]
    missing_index = ehr_missing.set_index("variable")
    missing_vars = [v for v in missing_vars if v in missing_index.index]
    missing = np.array(
        [
            pd.to_numeric(missing_index.loc[missing_vars, f"{center}_missing_rate"], errors="coerce") * 100
            for center in ["A", "B", "C"]
        ]
    ).T
    cmap_missing = cmap_with_na("YlOrRd")
    fig_b, ax_b = plt.subplots(figsize=(3.15, 3.55), facecolor="white")
    image_missing = heatmap(
        ax_b,
        missing,
        [label_of(v) for v in missing_vars],
        ["Center A", "Center B", "Center C"],
        cmap_missing,
        Normalize(0, 100),
        "{:.1f}",
        6.6,
    )
    cbar_missing = fig_b.colorbar(image_missing, ax=ax_b, orientation="horizontal", fraction=0.095, pad=0.12)
    cbar_missing.set_label("Missing rate (%)", fontsize=7, labelpad=2)
    cbar_missing.ax.tick_params(labelsize=6, length=2, pad=1)
    save(fig_b, out_dir / "RQ1_cross_center_heterogeneity_panel_b.png")

    # Panel C — strictly the six requested CT acquisition variables.
    ct_vars = [
        "spacing_z",
        "num_slices",
        "voxel_count",
        "inplane_resolution_mm",
        "axial_coverage_mm",
        "slice_thickness",
    ]
    ct_index = ct_shift.set_index("variable")
    ct_vars = [v for v in ct_vars if v in ct_index.index]
    ct_raw = np.array(
        [pd.to_numeric(ct_index.loc[ct_vars, f"{pair}_abs_SMD"], errors="coerce") for pair in pairs]
    ).T
    ct_for_color = np.log1p(ct_raw)
    cmap_ct = cmap_with_na("viridis")
    fig_c, ax_c = plt.subplots(figsize=(3.25, 2.95), facecolor="white")
    image_ct = heatmap(
        ax_c,
        ct_for_color,
        [label_of(v) for v in ct_vars],
        pair_labels,
        cmap_ct,
        Normalize(0, float(np.nanmax(ct_for_color))),
        "{:.2f}",
        6.4,
    )
    # Replace transformed display annotations with the original frozen |SMD| values.
    for text in list(ax_c.texts):
        text.remove()
    norm_ct = image_ct.norm
    for row in range(ct_raw.shape[0]):
        for column in range(ct_raw.shape[1]):
            raw_value, color_value = ct_raw[row, column], ct_for_color[row, column]
            if np.isnan(raw_value):
                ax_c.text(column, row, "NA", ha="center", va="center", fontsize=6.4, color="#5E5E5E", fontstyle="italic")
            else:
                ax_c.text(
                    column,
                    row,
                    f"{raw_value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color=annotation_color(cmap_ct, norm_ct, color_value),
                )
    cbar_ct = fig_c.colorbar(image_ct, ax=ax_c, orientation="horizontal", fraction=0.095, pad=0.12)
    cbar_ct.set_label("log1p(|SMD|); values in cells are original |SMD|", fontsize=6.3, labelpad=2)
    cbar_ct.ax.tick_params(labelsize=6, length=2, pad=1)
    save(fig_c, out_dir / "RQ1_cross_center_heterogeneity_panel_c.png")

    for panel in "abc":
        destination = out_dir / f"RQ1_cross_center_heterogeneity_panel_{panel}.png"
        if not destination.exists() or destination.stat().st_size == 0:
            raise RuntimeError(f"Missing output: {destination}")
        print(f"{destination} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
