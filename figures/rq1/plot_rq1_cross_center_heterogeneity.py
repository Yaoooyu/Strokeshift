from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Publication export settings: editable vector text and consistent sans-serif fonts.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 6.5
plt.rcParams["axes.titleweight"] = "bold"


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
    "diabetes_history": "Diabetes history",
    "dyslipidemia_history": "Dyslipidemia history",
    "smoking_history": "Smoking history",
    "alcohol_history": "Alcohol history",
    "prior_stroke_history": "Prior stroke history",
    "arterial_plaque": "Arterial plaque",
    "hypertension_drug": "Antihypertensive drug",
    "antiplatelet_drug": "Antiplatelet drug",
    "anticoagulant_drug": "Anticoagulant drug",
    "statin_drug": "Statin drug",
    "diabetes_drug": "Diabetes medication",
    "stent_implantation": "Stent implantation",
    "mechanical_thrombectomy": "Mechanical thrombectomy",
    "pre_stroke_mrs": "Pre-stroke mRS",
    "hd_CHD": "Coronary heart disease",
    "hd_MI": "Myocardial infarction",
    "hd_AF": "Atrial fibrillation",
    "hd_NVAF": "Non-valvular atrial fibrillation",
    "hd_VAF": "Valvular atrial fibrillation",
    "hd_Other_HD": "Other heart disease",
    "spacing_z": "Z-axis spacing",
    "num_slices": "Number of slices",
    "shape_z": "Z-axis image size",
    "voxel_count": "Voxel count",
    "inplane_resolution_mm": "In-plane resolution",
    "spacing_x": "X-axis spacing",
    "spacing_y": "Y-axis spacing",
    "axial_coverage_mm": "Axial coverage",
    "slice_thickness": "Slice thickness",
}


def label_of(name):
    return LABELS.get(name, name.replace("_", " ").replace("Hd ", ""))


def cmap_with_na(name):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("#E6E6E6")
    return cmap


def text_color(cmap, norm, value):
    rgba = cmap(norm(value))
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if luminance < 0.52 else "#202020"


def draw_heatmap(
    ax,
    matrix,
    row_labels,
    col_labels,
    cmap,
    norm,
    fmt,
    cbar_label,
    annotation_size=5.2,
    title=None,
    title_size=7.1,
    cbar_ticks=None,
    cbar_orientation="vertical",
    cbar_y=-0.20,
):
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=6.0)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=5.7)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.05)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=title_size, pad=16, fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                ax.text(
                    j,
                    i,
                    "NA",
                    ha="center",
                    va="center",
                    fontsize=annotation_size,
                    color="#5E5E5E",
                    fontstyle="italic",
                )
            else:
                ax.text(
                    j,
                    i,
                    fmt.format(value),
                    ha="center",
                    va="center",
                    fontsize=annotation_size,
                    color=text_color(cmap, norm, value),
                )
    if cbar_orientation == "horizontal":
        cax = inset_axes(
            ax,
            width="74%",
            height="5.5%",
            loc="lower left",
            bbox_to_anchor=(0.13, cbar_y, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = ax.figure.colorbar(image, cax=cax, ticks=cbar_ticks, orientation="horizontal")
        cbar.ax.tick_params(labelsize=5.0, length=2, pad=1)
        cbar.set_label(cbar_label, fontsize=5.6, labelpad=2)
    else:
        cax = inset_axes(
            ax,
            width="4.7%",
            height="85%",
            loc="lower left",
            bbox_to_anchor=(1.045, 0.075, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = ax.figure.colorbar(image, cax=cax, ticks=cbar_ticks)
        cbar.ax.tick_params(labelsize=5.2, length=2, pad=1)
        cbar.set_label(cbar_label, fontsize=5.8, labelpad=3)
    cbar.outline.set_linewidth(0.45)
    return image


def selected_rows(frame, variables):
    return frame.set_index("variable", drop=False).loc[variables].copy()


def main():
    root = Path(".")
    input_a = root / "analysis/RQ1/tables/figure_RQ1a_ehr_shift_corrected.csv"
    input_b = root / "analysis/RQ1/tables/figure_RQ1b_ehr_missingness_corrected.csv"
    input_c = root / "analysis/RQ1/tables/figure_RQ1c_ct_shift.csv"
    out_dir = root / "analysis/RQ1/figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only the three frozen, paper-facing input tables are read.
    ehr_shift = pd.read_csv(input_a)
    ehr_missing = pd.read_csv(input_b)
    ct_shift = pd.read_csv(input_c)

    continuous_priority = [
        "age",
        "admission_nihss",
        "admission_glucose",
        "dnt_time",
        "admission_sbp",
        "admission_dbp",
    ]
    categorical_priority = [
        "hypertension_drug",
        "dyslipidemia_history",
        "arterial_plaque",
        "hd_Other_HD",
        "hd_MI",
        "hypertension_history",
        "htn_history",
        "gender",
        "smoking_history",
        "alcohol_history",
        "prior_stroke_history",
    ]
    continuous_available = set(ehr_shift.loc[ehr_shift["variable_type"].eq("continuous"), "variable"])
    continuous_vars = [v for v in continuous_priority if v in continuous_available]
    continuous_vars += [
        v
        for v in ehr_shift.loc[ehr_shift["variable_type"].eq("continuous"), "variable"]
        if v not in continuous_vars
    ]
    categorical_available = set(ehr_shift.loc[ehr_shift["variable_type"].eq("categorical"), "variable"])
    categorical_vars = []
    for variable in categorical_priority:
        if variable in categorical_available and variable not in categorical_vars:
            categorical_vars.append(variable)
    if not categorical_vars:
        raise RuntimeError("No requested categorical variables found in corrected EHR shift table.")

    continuous_rows = selected_rows(ehr_shift, continuous_vars)
    categorical_rows = selected_rows(ehr_shift, categorical_vars)
    pairs = ["AB", "AC", "BC"]
    continuous_matrix = np.array(
        [
            np.abs(pd.to_numeric(continuous_rows[f"{pair}_SMD"], errors="coerce").to_numpy(float))
            for pair in pairs
        ]
    ).T
    categorical_matrix = np.array(
        [
            pd.to_numeric(categorical_rows[f"{pair}_effect_size"], errors="coerce").to_numpy(float)
            for pair in pairs
        ]
    ).T

    # All available EHR variables remain in Panel B, sorted by the supplied difference field.
    missing_rows = ehr_missing.sort_values("max_center_difference", ascending=False, kind="mergesort").copy()
    missing_matrix = np.column_stack(
        [
            pd.to_numeric(missing_rows[f"{center}_missing_rate"], errors="coerce").to_numpy(float) * 100
            for center in ["A", "B", "C"]
        ]
    )

    ct_requested = [
        "spacing_z",
        "num_slices",
        "shape_z",
        "voxel_count",
        "inplane_resolution_mm",
        "spacing_x",
        "spacing_y",
        "axial_coverage_mm",
        "slice_thickness",
    ]
    ct_indexed = ct_shift.set_index("variable", drop=False)
    ct_vars = [variable for variable in ct_requested if variable in ct_indexed.index]
    ct_rows = ct_indexed.loc[ct_vars].copy()
    ct_raw = np.array(
        [
            pd.to_numeric(ct_rows[f"{pair}_abs_SMD"], errors="coerce").to_numpy(float)
            for pair in pairs
        ]
    ).T
    # The specified transform controls color only; printed cell values stay raw |SMD|.
    ct_color = np.log1p(ct_raw)

    fig = plt.figure(figsize=(9.4, 6.8), facecolor="white")
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.45, 1.35, 1.25],
        left=0.105,
        right=0.99,
        bottom=0.19,
        top=0.91,
        wspace=0.54,
    )
    grid_a = outer[0].subgridspec(
        2,
        1,
        height_ratios=[len(continuous_vars), len(categorical_vars)],
        hspace=0.61,
    )
    ax_a_continuous = fig.add_subplot(grid_a[0])
    ax_a_categorical = fig.add_subplot(grid_a[1])
    ax_b = fig.add_subplot(outer[1])
    ax_c = fig.add_subplot(outer[2])

    cmap_smd = cmap_with_na("Blues")
    cmap_v = cmap_with_na("Purples")
    cmap_missing = cmap_with_na("YlOrRd")
    cmap_ct = cmap_with_na("viridis")
    finite_smd = continuous_matrix[np.isfinite(continuous_matrix)]
    finite_v = categorical_matrix[np.isfinite(categorical_matrix)]
    finite_ct_color = ct_color[np.isfinite(ct_color)]
    norm_smd = Normalize(vmin=0, vmax=max(0.60, float(finite_smd.max())))
    norm_v = Normalize(vmin=0, vmax=max(0.25, float(finite_v.max())))
    norm_missing = Normalize(vmin=0, vmax=100)
    norm_ct = Normalize(vmin=0, vmax=float(finite_ct_color.max()))

    draw_heatmap(
        ax_a_continuous,
        continuous_matrix,
        [label_of(variable) for variable in continuous_vars],
        ["A–B", "A–C", "B–C"],
        cmap_smd,
        norm_smd,
        "{:.2f}",
        "|SMD|",
        annotation_size=5.4,
        title="(a) Observed EHR Distribution Shift\nContinuous variables",
        title_size=7.8,
        cbar_orientation="horizontal",
    )
    draw_heatmap(
        ax_a_categorical,
        categorical_matrix,
        [label_of(variable) for variable in categorical_vars],
        ["A–B", "A–C", "B–C"],
        cmap_v,
        norm_v,
        "{:.2f}",
        "Cramér's V",
        annotation_size=5.25,
        title="Categorical variables (observed values only)",
        title_size=6.8,
        cbar_orientation="horizontal",
    )
    draw_heatmap(
        ax_b,
        missing_matrix,
        [label_of(variable) for variable in missing_rows["variable"]],
        ["Center A", "Center B", "Center C"],
        cmap_missing,
        norm_missing,
        "{:.1f}",
        "Missing rate (%)",
        annotation_size=4.55,
        title="(b) EHR Missingness\nAcross Centers",
        title_size=7.8,
        cbar_ticks=[0, 25, 50, 75, 100],
        cbar_orientation="horizontal",
        cbar_y=-0.11,
    )
    draw_heatmap(
        ax_c,
        ct_color,
        [label_of(variable) for variable in ct_vars],
        ["A–B", "A–C", "B–C"],
        cmap_ct,
        norm_ct,
        "{:.2f}",
        "log1p(|SMD|)",
        annotation_size=5.05,
        title_size=7.1,
        title="(c) Native CT\nAcquisition Shift",
    )
    # Replace transformed annotation values with original absolute SMD values.
    for text in list(ax_c.texts):
        text.remove()
    for i in range(ct_raw.shape[0]):
        for j in range(ct_raw.shape[1]):
            raw_value = ct_raw[i, j]
            color_value = ct_color[i, j]
            if np.isnan(raw_value):
                ax_c.text(
                    j,
                    i,
                    "NA",
                    ha="center",
                    va="center",
                    fontsize=5.05,
                    color="#5E5E5E",
                    fontstyle="italic",
                )
            else:
                ax_c.text(
                    j,
                    i,
                    f"{raw_value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.05,
                    color=text_color(cmap_ct, norm_ct, color_value),
                )

    ax_c.text(
        0,
        -0.12,
        "Mean acquisition |SMD|:\nA–C = 4.821, A–B = 3.787, B–C = 0.562.",
        transform=ax_c.transAxes,
        fontsize=5.45,
        color="#333333",
        ha="left",
        va="top",
        linespacing=1.32,
    )

    base = out_dir / "RQ1_cross_center_heterogeneity"
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(base.with_suffix(f".{extension}"), dpi=600, facecolor="white")
    plt.close(fig)
    for extension in ["pdf", "svg", "png"]:
        output = base.with_suffix(f".{extension}")
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"Failed export: {output}")
        print(f"{output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
