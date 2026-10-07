"""Build the paper figures from the final date-anchored unified cohort.

The script reads only aggregate, versioned outputs.  It owns the four main-figure
filenames and Supplementary Figure S1 so a later ``make figs`` cannot silently restore
the superseded visit-transition counts.  Every plotted data frame is also exported to
``paper/tables/figure_source_unified_*.csv``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


RUN = Path(__file__).resolve().parents[1]
FINAL_RUN = (
    RUN
    / "outputs/unified_cohort/runs/"
    / "semaglutide__floor_3m__stages_main__blocks_plus_lipids__modalities_paper__tabular"
)
SECONDARY = RUN / "outputs/unified_cohort/secondary_sleep"
ALTERNATIVE = RUN / "outputs/unified_cohort/alternative_matching"
ROBUST = RUN / "outputs/unified_cohort/sleep_robustness"
FIGURES = RUN / "paper/figures/final"
TABLES = RUN / "paper/tables"

# Okabe-Ito-derived, colorblind-safe semantic palette.
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
BLACK = "#222222"
GREY = "#777777"
LIGHT_GREY = "#D9D9D9"
PALE = "#F3F3F3"

TARGET_WIDTH_MM = 179.0
PAD_IN = 0.02

LABELS = {
    "blood_pressure": "Blood pressure",
    "sleep": "Sleep",
    "dxa": "Body composition (DXA)",
    "anthropometrics": "Anthropometrics",
    "hmo": "Blood tests",
    "microbiome": "Gut taxonomy",
    "microbiome_function": "Microbiome function",
    "diet": "Diet",
    "sitting_blood_pressure_systolic": "Systolic BP",
    "sitting_blood_pressure_diastolic": "Diastolic BP",
    "sitting_blood_pressure_pulse_rate": "Sitting pulse",
    "heart_rate_mean_during_sleep": "Whole-sleep HR",
    "heart_rate_mean_during_wake": "Nocturnal wake HR",
    "heart_rate_mean_during_nrem": "Non-REM HR",
    "heart_rate_mean_during_rem": "REM HR",
    "heart_rate_rem_minus_nrem": "REM − non-REM HR",
    "heart_rate_mean_during_light": "Light-sleep HR",
    "heart_rate_mean_during_deep": "Deep-sleep HR",
    "heart_rate_rem_minus_light": "REM − light HR",
    "heart_rate_rem_minus_deep": "REM − deep HR",
    "heart_rate_deep_minus_light": "Deep − light HR",
    "ahi": "AHI",
    "ahi_4_percent": "AHI (4% desaturation)",
    "ahi_during_rem": "REM AHI",
    "odi": "ODI",
    "odi_during_rem": "REM ODI",
    "rdi_during_rem": "REM RDI",
    "sleep_efficiency": "Sleep efficiency",
    "neurokit_hrv_time_pnn20_during_rem": "REM pNN20",
    "neurokit_hrv_time_rmssd_during_rem": "REM RMSSD",
    "neurokit_hrv_time_sdnn_during_rem": "REM SDNN",
    "neurokit_hrv_time_pnn20_during_nrem": "Non-REM pNN20",
    "neurokit_hrv_time_rmssd_during_nrem": "Non-REM RMSSD",
    "neurokit_hrv_time_sdnn_during_nrem": "Non-REM SDNN",
    "body_comp_android_fat_mass": "Android fat mass",
    "body_comp_gynoid_fat_mass": "Gynoid fat mass",
    "body_comp_total_fat_mass": "Total fat mass",
    "body_comp_total_lean_mass": "Total lean mass",
    "total_scan_vat_area": "Visceral fat area",
    "total_scan_vat_volume": "Visceral fat volume",
    "bt_ldl_cholesterol": "LDL cholesterol",
    "bt_non_hdl_cholesterol": "Non-HDL cholesterol",
    "bt_hba1c": "HbA1c",
    "energy_kcal_per_day": "Energy intake",
    "pct_fat_calories": "Dietary fat share",
    "pct_carb_calories": "Dietary carbohydrate share",
    "pct_protein_calories": "Dietary protein share",
    "microbiome_clr::Desulfovibrio": "Desulfovibrio",
    "microbiome_clr::Blautia": "Blautia",
    "microbiome_clr::Bifidobacterium": "Bifidobacterium",
    "microbiome_clr::Dorea": "Dorea",
    "microbiome_clr::Megasphaera": "Megasphaera",
    "microbiome_clr::Roseburia": "Roseburia",
    "microbiome_shannon": "Shannon diversity",
    "weight": "Weight",
    "age": "Age",
    "bmi": "BMI",
    "waist_circumference": "Waist circumference",
    "hip_circumference": "Hip circumference",
    "vat_area": "Visceral fat area",
    "vat_missing": "Visceral fat not measured",
    "gender": "Sex",
    "diabetic": "Diabetes",
    "smoking": "Current smoking",
    "on_antihypertensive": "Antihypertensive\ntherapy",
    "on_lipid_lowering": "Lipid-lowering\ntherapy",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def label(value: str) -> str:
    return LABELS.get(value, value.replace("_", " ").capitalize())


def clean_axis(ax: plt.Axes, grid: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color="#ECECEC", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, text: str, x: float = -0.12) -> None:
    """Letter on the title's line, whatever the axes height, so letters align across panels."""
    ax.annotate(text, xy=(x, 1), xycoords="axes fraction",
                xytext=(0, plt.rcParams["axes.titlepad"]), textcoords="offset points",
                ha="left", va="bottom", fontsize=9, fontweight="bold")


def fit_to_width(fig: plt.Figure, target_mm: float = TARGET_WIDTH_MM) -> float:
    target = target_mm / 25.4 - 2 * PAD_IN
    for _ in range(8):
        fig.canvas.draw()
        width = fig.get_tightbbox(fig.canvas.get_renderer()).width
        if abs(width - target) < 0.001:
            break
        current_w, current_h = fig.get_size_inches()
        fig.set_size_inches(current_w * target / width, current_h)
    fig.canvas.draw()
    return fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * PAD_IN


def save(fig: plt.Figure, stem: str, target_mm: float = TARGET_WIDTH_MM) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    width = fit_to_width(fig, target_mm)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", pad_inches=PAD_IN)
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=PAD_IN)
    plt.close(fig)
    print(f"wrote {stem}.pdf/.png ({width * 25.4:.0f} mm)")


def forest(ax: plt.Axes, frame: pd.DataFrame, *, y: np.ndarray | None = None,
           effect: str = "effect", low: str = "ci_low", high: str = "ci_high",
           colours: list[str] | None = None, markers: list[str] | None = None) -> None:
    if y is None:
        y = np.arange(len(frame))
    colours = colours or [VERMILLION] * len(frame)
    markers = markers or ["o"] * len(frame)
    for i, (_, row) in enumerate(frame.iterrows()):
        ax.errorbar(
            row[effect], y[i],
            xerr=np.array([[row[effect] - row[low]], [row[high] - row[effect]]]),
            fmt=markers[i], color=colours[i], ecolor=colours[i], markersize=4,
            elinewidth=1.05, capsize=2, zorder=3,
        )


# steps/51_unified_sleep_robustness.py outputs, keyed as the figure functions read them.
ROBUST_FILES = {
    "exposure_definitions": "exposure_definitions",
    "design_estimates": "design_estimates",
    "rematched": "rematched_bootstrap_summary",
    "drift": "untreated_drift",
    "scale_per_pair": "relative_scale_per_pair",
    "scale_contrasts": "relative_scale_contrasts",
    "baseline_dependence": "baseline_dependence",
    "systemic": "systemic_rise_adjustment",
}


def load_inputs() -> dict[str, pd.DataFrame]:
    required = [
        FINAL_RUN / "run_manifest.json",
        FINAL_RUN / "primary_results.csv",
        FINAL_RUN / "modality_status.csv",
        FINAL_RUN / "balance_sleep.csv",
        FINAL_RUN / "funnel.csv",
        FINAL_RUN / "timing_diagnostics.csv",
        SECONDARY / "hrv_rate_adjustment.csv",
        SECONDARY / "raw_stage_results.csv",
        SECONDARY / "raw_stage_design.csv",
        ALTERNATIVE / "endpoint_estimates.csv",
        ALTERNATIVE / "design_diagnostics.csv",
        ALTERNATIVE / "design_selection.csv",
        ALTERNATIVE / "posthoc_selected_hierarchical_results.csv",
        *(ROBUST / f"{name}.csv" for name in ROBUST_FILES.values()),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing unified figure inputs:\n" + "\n".join(missing))
    manifest = json.loads((FINAL_RUN / "run_manifest.json").read_text())
    if manifest["dated_clean_initiators"] != 494:
        raise AssertionError("The final figure builder is not reading the reviewed 494-person run")
    return {
        "primary": pd.read_csv(FINAL_RUN / "primary_results.csv"),
        "status": pd.read_csv(FINAL_RUN / "modality_status.csv"),
        "balance": pd.read_csv(FINAL_RUN / "balance_sleep.csv"),
        "funnel": pd.read_csv(FINAL_RUN / "funnel.csv"),
        "timing": pd.read_csv(FINAL_RUN / "timing_diagnostics.csv"),
        "hrv": pd.read_csv(SECONDARY / "hrv_rate_adjustment.csv"),
        "raw": pd.read_csv(SECONDARY / "raw_stage_results.csv"),
        "raw_design": pd.read_csv(SECONDARY / "raw_stage_design.csv"),
        "alternative": pd.read_csv(ALTERNATIVE / "endpoint_estimates.csv"),
        "alternative_design": pd.read_csv(ALTERNATIVE / "design_diagnostics.csv"),
        "alternative_selection": pd.read_csv(ALTERNATIVE / "design_selection.csv"),
        "alternative_hierarchical": pd.read_csv(
            ALTERNATIVE / "posthoc_selected_hierarchical_results.csv"
        ),
        **{key: pd.read_csv(ROBUST / f"{name}.csv") for key, name in ROBUST_FILES.items()},
    }


def figure1_sources(data: dict[str, pd.DataFrame]) -> None:
    """Export the rows Figure 1d plots; the figure itself is built by `build_fig1_unified.py`.

    Panel d reads the same sleep rows of the locked run, so writing them here keeps the
    figure's source table beside the others and current with every rerun.
    """
    sleep = data["primary"][data["primary"].modality.eq("sleep")].set_index("measurement")
    states = sleep.loc[[
        "heart_rate_mean_during_sleep", "heart_rate_mean_during_wake",
        "heart_rate_mean_during_nrem", "heart_rate_mean_during_rem",
    ]].reset_index()
    TABLES.mkdir(parents=True, exist_ok=True)
    states.to_csv(TABLES / "figure_source_unified_fig1_states.csv", index=False)
    sleep.loc[["heart_rate_rem_minus_nrem"]].reset_index().to_csv(
        TABLES / "figure_source_unified_fig1_contrast.csv", index=False)


def figure2(data: dict[str, pd.DataFrame]) -> None:
    primary = data["primary"].copy()
    baseline = primary[[
        "modality", "measurement", "equivalence_margin_0p2_baseline_sd"
    ]].copy()
    baseline["baseline_sd"] = baseline.equivalence_margin_0p2_baseline_sd / 0.2
    if baseline.baseline_sd.isna().any() or baseline.baseline_sd.le(0).any():
        raise AssertionError("every Figure 2 outcome needs a positive primary baseline SD")

    claims = primary[primary.claim_eligible.astype(bool)].copy().merge(
        baseline[["modality", "measurement", "baseline_sd"]],
        on=["modality", "measurement"], how="left", validate="one_to_one",
    )
    claims = claims.rename(columns={"n_pairs": "n"})
    claims["design"] = "primary_optimal_1to1"
    claims["evidence_class"] = "primary_claim"
    claims["posthoc_hierarchical_significant"] = True
    claims["endpoint_balance_pass"] = True

    hmo_measurements = [
        "bt_hba1c", "bt_ldl_cholesterol", "bt_non_hdl_cholesterol",
    ]
    unresolved = primary[
        primary.modality.eq("hmo") & primary.measurement.isin(hmo_measurements)
    ].copy().merge(
        baseline[["modality", "measurement", "baseline_sd"]],
        on=["modality", "measurement"], how="left", validate="one_to_one",
    )
    unresolved = unresolved.rename(columns={"n_pairs": "n"})
    unresolved["design"] = "primary_optimal_1to1"
    unresolved["evidence_class"] = "unresolved_balance"
    unresolved["posthoc_hierarchical_significant"] = False
    unresolved["endpoint_balance_pass"] = False

    # Weighted rows come from every modality whose 1:1 match failed balance in the run; blood
    # tests drop out here because none of their outcome-complete samples balances.
    status = data["status"]
    failed = status.loc[status.status.eq("estimated")
                        & ~status.design_gate_pass.astype(bool), "modality"]
    alternatives = data["alternative"].copy()
    alternatives = alternatives[
        alternatives.design.eq("exact_stratum_overlap")
        & alternatives.modality.isin(failed)
        & alternatives.endpoint_balance_pass.astype(bool)
    ].merge(
        baseline[["modality", "measurement", "baseline_sd"]],
        on=["modality", "measurement"], how="left", validate="one_to_one",
    )
    alternatives = alternatives.rename(columns={"n_treated": "n"})
    hierarchical = data["alternative_hierarchical"][[
        "modality", "measurement", "posthoc_modality_gate_fdr",
        "posthoc_hierarchical_significant",
    ]]
    alternatives = alternatives.merge(
        hierarchical,
        on=["modality", "measurement"], how="left", validate="one_to_one",
    )
    alternatives["posthoc_hierarchical_significant"] = (
        alternatives.posthoc_hierarchical_significant.fillna(False).astype(bool)
    )
    alternatives["evidence_class"] = np.where(
        alternatives.posthoc_hierarchical_significant,
        "weighted_hierarchical", "weighted_null",
    )

    common = [
        "modality", "measurement", "n", "effect", "ci_low", "ci_high",
        "baseline_sd", "design", "evidence_class", "endpoint_balance_pass",
        "posthoc_hierarchical_significant",
    ]
    selected = pd.concat(
        [claims[common], unresolved[common], alternatives[common]], ignore_index=True,
    )
    selected["std_effect"] = selected.effect / selected.baseline_sd
    selected["std_low"] = selected.ci_low / selected.baseline_sd
    selected["std_high"] = selected.ci_high / selected.baseline_sd
    selected["standardizer"] = "baseline SD from locked primary 1:1 analysis"
    expected = {
        "primary_claim": 23,
        "weighted_null": 1,
        "weighted_hierarchical": 7,
        "unresolved_balance": 3,
    }
    observed = selected.evidence_class.value_counts().to_dict()
    if observed != expected:
        raise AssertionError(f"unexpected Figure 2 evidence counts: {observed}")

    order = [
        "sleep", "blood_pressure", "dxa", "anthropometrics", "microbiome", "diet",
        "hmo",
    ]
    selected["modality_order"] = selected.modality.map(
        {modality: i for i, modality in enumerate(order)}
    )
    selected = selected.sort_values(["modality_order", "std_effect"])

    fig = plt.figure(figsize=(7.2, 10.25))
    outer = fig.add_gridspec(
        1, 2, width_ratios=[1.48, 0.90], wspace=0.60,
        left=0.12, right=0.98, top=0.965, bottom=0.095,
    )
    ax = fig.add_subplot(outer[0, 0])
    rows, ticklabels, spans = [], [], []
    cursor = 0
    for modality in order:
        block = selected[selected.modality.eq(modality)]
        if block.empty:
            continue
        start = cursor
        rows.append(None)
        ticklabels.append(label(modality))
        cursor += 1
        for _, row in block.iterrows():
            rows.append(row)
            ticklabels.append(f"   {label(row.measurement)} (n={int(row.n)})")
            cursor += 1
        spans.append((start - 0.5, cursor - 0.5))
    for block_i, (low, high) in enumerate(spans):
        if block_i % 2:
            ax.axhspan(low, high, color=PALE, zorder=0)
    for y, row in enumerate(rows):
        if row is None:
            continue
        evidence = row.evidence_class
        if evidence == "primary_claim":
            colour, marker, face = VERMILLION, "o", VERMILLION
        elif evidence == "weighted_hierarchical":
            colour, marker, face = PURPLE, "D", PURPLE
        elif evidence == "weighted_null":
            colour, marker, face = PURPLE, "D", "white"
        else:
            colour, marker, face = GREY, "X", GREY
        ax.errorbar(
            row.std_effect, y,
            xerr=np.array([[row.std_effect - row.std_low],
                           [row.std_high - row.std_effect]]),
            fmt=marker, color=colour, ecolor=colour,
            markerfacecolor=face, markeredgecolor=colour, markeredgewidth=0.9,
            capsize=2, markersize=3.8, elinewidth=1.0, zorder=3,
        )
    ax.axvline(0, color=BLACK, linewidth=0.8, linestyle="--")
    ax.set_yticks(np.arange(len(rows)), ticklabels)
    for tick, row in zip(ax.get_yticklabels(), rows, strict=True):
        if row is None:
            tick.set_fontweight("bold")
            tick.set_fontsize(6.7)
        else:
            tick.set_fontsize(6.2)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel("Standardized DiD")
    clean_axis(ax)
    ax.legend(handles=[
        Line2D([], [], marker="o", color=VERMILLION, linestyle="none",
               label="1:1 matched comparison, hierarchical q<0.05"),
        Line2D([], [], marker="D", color=PURPLE, linestyle="none",
               label="Weighted comparison, hierarchical q<0.05"),
        Line2D([], [], marker="D", markerfacecolor="white", markeredgecolor=PURPLE,
               color=PURPLE, linestyle="none",
               label="Weighted comparison, not significant"),
        Line2D([], [], marker="X", color=GREY, linestyle="none",
               label="Covariate imbalance remains"),
    ], frameon=False, loc="upper center", bbox_to_anchor=(0.50, -0.035),
       ncol=1, fontsize=6.3)
    panel_label(ax, "a", x=-0.43)

    right = outer[0, 1].subgridspec(2, 1, height_ratios=[0.96, 1.04], hspace=0.50)
    ax = fig.add_subplot(right[0, 0])
    status = data["status"].copy()
    overlap = data["alternative_design"][
        data["alternative_design"].design.eq("exact_stratum_overlap")
    ][["modality", "max_abs_smd"]].rename(columns={"max_abs_smd": "overlap_smd"})
    choices = data["alternative_selection"][[
        "modality", "n_estimable_endpoints", "n_endpoint_balance_pass",
    ]]
    status = status.merge(overlap, on="modality", how="left").merge(
        choices, on="modality", how="left"
    )
    status["display"] = status.modality.map(label)
    status = status.sort_values("max_abs_smd", na_position="last").reset_index(drop=True)
    y = np.arange(len(status))
    for i, row in status.iterrows():
        primary_colour = GREEN if bool(row.design_gate_pass) else GREY
        if pd.notna(row.overlap_smd):
            ax.plot([row.max_abs_smd, row.overlap_smd], [i, i],
                    color=LIGHT_GREY, linewidth=1.0, zorder=1)
        else:
            ax.plot([0, row.max_abs_smd], [i, i],
                    color=LIGHT_GREY, linewidth=0.8, zorder=1)
        ax.scatter(row.max_abs_smd, i, color=primary_colour, marker="o", s=22, zorder=3)
        if pd.notna(row.overlap_smd):
            face = PURPLE if int(row.n_endpoint_balance_pass) > 0 else "white"
            ax.scatter(row.overlap_smd, i, facecolor=face, edgecolor=PURPLE,
                       marker="D", linewidth=1.0, s=25, zorder=4)
        ax.text(
            1.02, i, f"{int(row.matched)} pairs", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=5.9, color=GREY, clip_on=False,
        )
    ax.axvline(0.10, color=BLACK, linestyle="--", linewidth=0.8)
    ax.set_yticks(y, status.display)
    ax.invert_yaxis()
    ax.set_xlim(-0.015, max(0.34, float(status.max_abs_smd.max()) * 1.05))
    ax.set_xlabel("Maximum absolute SMD")
    clean_axis(ax)
    ax.legend(handles=[
        Line2D([], [], marker="o", color=GREEN, linestyle="none",
               label="1:1 matched, balance passed"),
        Line2D([], [], marker="o", color=GREY, linestyle="none",
               label="1:1 matched, balance failed"),
        Line2D([], [], marker="D", color=PURPLE, linestyle="none", label="Weighted"),
        Line2D([], [], marker="D", markerfacecolor="white", markeredgecolor=PURPLE,
               linestyle="none", label="Weighted, no outcome-complete\nsample balanced"),
    ], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15),
       ncol=1, fontsize=6.2)
    panel_label(ax, "b", x=-0.50)

    ax = fig.add_subplot(right[1, 0])
    balance = data["balance"].copy()
    balance = balance[~balance.covariate.eq("vat_missing")]
    balance["before"] = balance.smd_before.abs()
    balance["after"] = balance.smd_after.abs()
    balance = balance.sort_values("after")
    y = np.arange(len(balance))
    for i, row in enumerate(balance.itertuples(index=False)):
        ax.plot([row.before, row.after], [i, i], color=LIGHT_GREY, linewidth=0.8)
        ax.scatter(row.before, i, facecolor="white", edgecolor=GREY, s=19, zorder=3)
        ax.scatter(row.after, i, color=BLUE, s=19, zorder=3)
    ax.axvline(0.10, color=BLACK, linestyle="--", linewidth=0.8)
    ax.set_yticks(y, balance.covariate.map(label))
    ax.invert_yaxis()
    ax.set_xlabel("Absolute standardized mean difference")
    clean_axis(ax)
    ax.legend(handles=[
        Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=GREY,
               linestyle="none", label="Before matching"),
        Line2D([], [], marker="o", color=BLUE, linestyle="none", label="After matching"),
    ], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13),
       ncol=1, fontsize=6.2)
    panel_label(ax, "c", x=-0.50)
    selected.to_csv(TABLES / "figure_source_unified_fig2_claims.csv", index=False)
    status.to_csv(TABLES / "figure_source_unified_fig2_design.csv", index=False)
    balance.to_csv(TABLES / "figure_source_unified_fig2_balance.csv", index=False)
    save(fig, "fig2_clinical_validation")


def figure4_hrv(data: dict[str, pd.DataFrame]) -> None:
    primary = data["primary"].set_index("measurement")
    models = data["hrv"].copy()
    models["measurement"] = models.model.str.extract(
        r"sleep:(neurokit_hrv_time_[^ ]+) adjusted", expand=False)
    models = models.merge(
        primary[["effect", "ci_low", "ci_high", "equivalence_margin_0p2_baseline_sd"]],
        left_on="measurement", right_index=True, how="left",
    )
    order = [
        "neurokit_hrv_time_pnn20_during_rem",
        "neurokit_hrv_time_sdnn_during_rem",
        "neurokit_hrv_time_rmssd_during_rem",
        "neurokit_hrv_time_pnn20_during_nrem",
        "neurokit_hrv_time_sdnn_during_nrem",
        "neurokit_hrv_time_rmssd_during_nrem",
    ]
    models = models.set_index("measurement").loc[order].reset_index()
    models["baseline_sd"] = models.equivalence_margin_0p2_baseline_sd / 0.2
    fig, ax = plt.subplots(figsize=(4.7, 3.5))
    y = np.arange(len(models))
    for effect, low, high, offset, colour, marker, name in [
        ("effect", "ci_low", "ci_high", -0.11, GREY, "o", "Unadjusted"),
        ("covariate_adjusted_intercept", "intercept_ci_low", "intercept_ci_high",
         +0.11, GREEN, "s", "Adjusted for simultaneous state HR change"),
    ]:
        plot = models.copy()
        for column in (effect, low, high):
            plot[column] = plot[column] / plot.baseline_sd
        forest(ax, plot, y=y + offset, effect=effect, low=low, high=high,
               colours=[colour] * len(plot), markers=[marker] * len(plot))
    ax.axvline(0, color=BLACK, linestyle="--", linewidth=0.8)
    ax.set_yticks(y, [f"{label(m)} (n={int(n)})" for m, n in
                      zip(models.measurement, models.n, strict=True)])
    ax.invert_yaxis()
    ax.set_xlabel("HRV DiD (baseline SD)")
    clean_axis(ax)
    ax.legend(handles=[
        Line2D([], [], marker="o", color=GREY, linestyle="none", label="Unadjusted"),
        Line2D([], [], marker="s", color=GREEN, linestyle="none",
               label="Heart-rate adjusted"),
    ], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    models.to_csv(TABLES / "figure_source_unified_fig4_hrv.csv", index=False)
    save(fig, "fig4_hrv_rate_dependence", target_mm=118.0)


def figure_s6_stage(data: dict[str, pd.DataFrame]) -> None:
    raw = data["raw"].set_index("measurement")
    stages = raw.loc[[
        "heart_rate_mean_during_rem", "heart_rate_mean_during_light",
        "heart_rate_mean_during_deep", "heart_rate_mean_during_wake",
    ]].reset_index()
    contrasts = raw.loc[[
        "heart_rate_rem_minus_nrem", "heart_rate_rem_minus_light",
        "heart_rate_rem_minus_deep", "heart_rate_deep_minus_light",
    ]].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    fig.subplots_adjust(left=0.13, right=0.985, top=0.92, bottom=0.20, wspace=0.88)
    for ax, frame, letter in [(axes[0], stages, "a"), (axes[1], contrasts, "b")]:
        colours = [PURPLE if low > 0 or high < 0 else GREY
                   for low, high in zip(frame.ci_low, frame.ci_high, strict=True)]
        forest(ax, frame, colours=colours)
        ax.axvline(0, color=BLACK, linestyle="--", linewidth=0.8)
        short = {
            "heart_rate_mean_during_rem": "REM",
            "heart_rate_mean_during_light": "Light sleep",
            "heart_rate_mean_during_deep": "Deep sleep",
            "heart_rate_mean_during_wake": "Nocturnal wake",
            "heart_rate_rem_minus_nrem": "REM − non-REM",
            "heart_rate_rem_minus_light": "REM − light",
            "heart_rate_rem_minus_deep": "REM − deep",
            "heart_rate_deep_minus_light": "Deep − light",
        }
        ax.set_yticks(np.arange(len(frame)),
                      [f"{short[m]} (n={int(n)})" for m, n in
                       zip(frame.measurement, frame.n_pairs, strict=True)])
        ax.invert_yaxis()
        ax.set_xlabel("DiD (bpm)")
        clean_axis(ax)
        panel_label(ax, letter, x=-0.55)
    raw.reset_index().to_csv(TABLES / "figure_source_unified_figS6_raw.csv", index=False)
    save(fig, "figS6_stage_resolved_heart_rate")


def figure_s1(data: dict[str, pd.DataFrame]) -> None:
    primary = data["primary"]
    funnel = data["funnel"]
    sleep = funnel[funnel.instrument.eq("sleep")].set_index("stage").n
    counts = [494, int(sleep.iloc[1]), int(sleep.iloc[2]), int(sleep.iloc[3]),
              int(primary.loc[primary.measurement.eq("heart_rate_rem_minus_nrem"),
                              "n_pairs"].iloc[0])]
    labels = [
        "semaglutide initiators with a plausible recorded date",
        "with a sleep visit before and ≥3 months after the recorded start",
        "with complete matching covariates and at least one sleep outcome",
        "matched 1:1 without control reuse",
        "with both REM and non-REM heart rate at both visits",
    ]
    reasons = [
        "no sleep study before and ≥3 months after the start",
        "outcome/covariate incomplete",
        "outside propensity caliper",
        "REM or non-REM value unavailable",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.1, 5.2)
    ax.axis("off")
    ys = np.linspace(4.75, 0.55, len(counts))
    for i, (y, n, text) in enumerate(zip(ys, counts, labels, strict=True)):
        ax.add_patch(FancyBboxPatch((0.15, y - 0.30), 6.25, 0.60,
                                    boxstyle="round,pad=0.03", facecolor="white",
                                    edgecolor=BLACK, linewidth=0.9))
        ax.text(0.42, y, f"n = {n}", ha="left", va="center", fontweight="bold")
        ax.text(1.48, y, text, ha="left", va="center", fontsize=7)
        if i:
            lost = counts[i - 1] - n
            ax.add_patch(FancyArrowPatch((0.92, ys[i - 1] - 0.33), (0.92, y + 0.33),
                                         arrowstyle="-|>", mutation_scale=8,
                                         linewidth=0.8, color=GREY))
            ax.text(1.32, (ys[i - 1] + y) / 2, f"−{lost}", ha="left", va="center",
                    fontsize=7, fontweight="bold", color=GREY)
            ax.text(2.08, (ys[i - 1] + y) / 2, reasons[i - 1], ha="left", va="center",
                    fontsize=6.6, color=GREY)
    ax.text(6.85, 4.70, "Primary matching specification", fontweight="bold", fontsize=8)
    ax.text(6.85, 4.28,
            "Exact blocks\n• pre visit\n• post visit\n• sex\n• lipid-lowering therapy\n\n"
            "Propensity + Mahalanobis\n• age, BMI, waist, hip\n• visceral fat (+ missingness)\n"
            "• diabetes, smoking\n• antihypertensive therapy",
            ha="left", va="top", fontsize=7, linespacing=1.38)
    gate = data["status"].set_index("modality").loc["sleep"]
    if not bool(gate.design_gate_pass):
        raise AssertionError("Figure S1 describes a sleep match that passed its design gates")
    ax.text(6.85, 1.22,
            "Sleep design checks\n"
            f"retention: {counts[3]}/{counts[2]} = {counts[3] / counts[2]:.1%}\n"
            f"max |SMD|: {gate.max_abs_smd:.3f}\n"
            "both checks passed",
            ha="left", va="top", fontsize=7, color=GREEN, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": GREEN, "linewidth": 0.9})
    pd.DataFrame({"stage": labels, "n": counts}).to_csv(
        TABLES / "figure_source_unified_figS1_funnel.csv", index=False)
    save(fig, "figS1_study_design")


def baseline_sd(primary: pd.DataFrame, measurement: str) -> float:
    """Baseline SD of an endpoint, as recorded by the primary analysis."""
    return float(primary.set_index("measurement").loc[
        measurement, "equivalence_margin_0p2_baseline_sd"]) / 0.2


def interval_point(ax: plt.Axes, x: float, low: float, high: float, y: float, *,
                   colour: str, marker: str, filled: bool, size: float = 4.0,
                   horizontal: bool = True) -> None:
    """One estimate with its interval; open markers mark a failed design gate."""
    error = np.array([[x - low], [high - x]])
    ax.errorbar(
        x if horizontal else y, y if horizontal else x,
        xerr=error if horizontal else None, yerr=None if horizontal else error,
        fmt=marker, color=colour, ecolor=colour, markersize=size,
        markerfacecolor=colour if filled else "white", markeredgecolor=colour,
        markeredgewidth=0.9, elinewidth=1.0, capsize=2, zorder=3,
    )


EXPOSURE_ROWS = [
    ("semaglutide", "Semaglutide\n(primary cohort)"),
    ("any_glp1", "Any GLP-1\nreceptor agonist"),
    ("non_semaglutide", "Liraglutide, dulaglutide\nor tirzepatide"),
]
DESIGN_STYLE = {
    "optimal_1to1": ("1:1 matched", VERMILLION, "o"),
    "primary_optimal_1to1": ("1:1 matched, no reuse (primary)", VERMILLION, "o"),
    "capped_1to3_reuse2": ("1:1 to 1:3 matched, each control used ≤2 times", GREEN, "s"),
    "exact_stratum_overlap": ("Overlap weighted", PURPLE, "D"),
}


def figure_s5_exposure(data: dict[str, pd.DataFrame]) -> None:
    frame = data["exposure_definitions"]
    panels = [
        ("weight", "Weight", "kg"),
        ("sitting_blood_pressure_systolic", "Systolic blood pressure", "mmHg"),
        ("heart_rate_mean_during_rem", "REM sleep heart rate", "bpm"),
        ("heart_rate_rem_minus_nrem", "REM − non-REM contrast", "bpm"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharey=True,
                             gridspec_kw={"wspace": 0.10, "hspace": 0.62})
    offsets = {"optimal_1to1": -0.15, "exact_stratum_overlap": 0.15}
    for panel_i, (ax, (measurement, title, unit)) in enumerate(
            zip(axes.flat, panels, strict=True)):
        work = frame[frame.measurement.eq(measurement)]
        for y, (exposure, _label) in enumerate(EXPOSURE_ROWS):
            for row in work[work.exposure.eq(exposure)].itertuples(index=False):
                _name, colour, marker = DESIGN_STYLE[row.design]
                interval_point(ax, row.effect, row.ci_low, row.ci_high,
                               y + offsets[row.design], colour=colour, marker=marker,
                               filled=bool(row.gate_pass))
                ax.annotate(f"{int(row.n_treated)}", xy=(row.ci_high, y + offsets[row.design]),
                            xytext=(3, 0), textcoords="offset points", ha="left",
                            va="center", fontsize=5.6, color=GREY)
        ax.axvline(0, color=BLACK, linestyle="--", linewidth=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel(f"DiD ({unit})")
        ax.margins(x=0.12)
        clean_axis(ax)
        panel_label(ax, "abcd"[panel_i], x=-0.40 if panel_i % 2 == 0 else -0.06)
    axes[0, 0].set_yticks(range(len(EXPOSURE_ROWS)),
                          [label for _key, label in EXPOSURE_ROWS])
    axes[0, 0].set_ylim(len(EXPOSURE_ROWS) - 0.5, -0.5)
    fig.legend(handles=[
        Line2D([], [], marker="o", color=VERMILLION, linestyle="none", label="1:1 matched"),
        Line2D([], [], marker="D", color=PURPLE, linestyle="none", label="Overlap weighted"),
        Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=GREY,
               linestyle="none", label="Open: design check failed"),
    ], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    frame.to_csv(TABLES / "figure_source_unified_figS5_exposure.csv", index=False)
    save(fig, "figS5_exposure_definitions")


def figure_s2_arms(data: dict[str, pd.DataFrame]) -> None:
    primary = data["primary"].copy()
    targets = [
        "sitting_blood_pressure_systolic", "sitting_blood_pressure_diastolic",
        "sitting_blood_pressure_pulse_rate", "heart_rate_mean_during_sleep",
        "heart_rate_mean_during_wake", "heart_rate_mean_during_nrem",
        "heart_rate_mean_during_rem", "body_comp_total_fat_mass",
        "body_comp_total_lean_mass", "total_scan_vat_area",
    ]
    work = primary[primary.measurement.isin(targets)].set_index("measurement").loc[targets].reset_index()
    work["baseline_sd"] = work.equivalence_margin_0p2_baseline_sd / 0.2
    for column in ["treated_delta", "treated_delta_ci_low", "treated_delta_ci_high",
                   "control_delta", "control_delta_ci_low", "control_delta_ci_high"]:
        work[f"std_{column}"] = work[column] / work.baseline_sd
    drift = data["drift"].set_index("measurement").loc[targets].reset_index()
    drift["baseline_sd"] = work.baseline_sd.to_numpy()
    for column in ["drift", "drift_ci_low", "drift_ci_high"]:
        drift[f"std_{column}"] = drift[column] / drift.baseline_sd

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True,
                             gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.08})
    y = np.arange(len(work))
    ax = axes[0]
    for effect, low, high, offset, colour, marker in [
        ("std_control_delta", "std_control_delta_ci_low", "std_control_delta_ci_high",
         -0.11, GREY, "s"),
        ("std_treated_delta", "std_treated_delta_ci_low", "std_treated_delta_ci_high",
         +0.11, VERMILLION, "o"),
    ]:
        forest(ax, work, y=y + offset, effect=effect, low=low, high=high,
               colours=[colour] * len(work), markers=[marker] * len(work))
    ax.axvline(0, color=BLACK, linestyle="--", linewidth=0.8)
    ax.set_yticks(y, [f"{label(m)} (n={int(n)})" for m, n in
                      zip(work.measurement, work.n_pairs, strict=True)])
    ax.invert_yaxis()
    ax.set_xlabel("Within-person change (baseline SD)")
    ax.set_title("Change in each group", loc="left")
    clean_axis(ax)
    ax.legend(handles=[
        Line2D([], [], marker="o", color=VERMILLION, linestyle="none",
               label="Semaglutide initiators"),
        Line2D([], [], marker="s", color=GREY, linestyle="none", label="Matched controls"),
    ], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    panel_label(ax, "a", x=-0.62)

    ax = axes[1]
    forest(ax, work, y=y - 0.11, effect="std_control_delta",
           low="std_control_delta_ci_low", high="std_control_delta_ci_high",
           colours=[GREY] * len(work), markers=["s"] * len(work))
    forest(ax, drift, y=y + 0.11, effect="std_drift", low="std_drift_ci_low",
           high="std_drift_ci_high", colours=[SKY] * len(drift), markers=["D"] * len(drift))
    ax.axvline(0, color=BLACK, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Within-person change (baseline SD)")
    ax.set_title("Controls vs untreated cohort", loc="left")
    clean_axis(ax)
    ax.legend(handles=[
        Line2D([], [], marker="s", color=GREY, linestyle="none", label="Matched controls"),
        Line2D([], [], marker="D", color=SKY, linestyle="none",
               label="All never-GLP-1 participants"),
    ], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    panel_label(ax, "b", x=-0.10)
    work.to_csv(TABLES / "figure_source_unified_figS2_arms.csv", index=False)
    drift.to_csv(TABLES / "figure_source_unified_figS2_drift.csv", index=False)
    save(fig, "figS2_arm_trajectories")


def figure_s4_designs(data: dict[str, pd.DataFrame]) -> None:
    primary = data["primary"]
    designs = data["design_estimates"].copy()
    endpoints = [
        "weight", "sitting_blood_pressure_systolic", "sitting_blood_pressure_pulse_rate",
        "heart_rate_mean_during_rem", "heart_rate_rem_minus_nrem",
    ]
    designs["baseline_sd"] = designs.measurement.map(lambda m: baseline_sd(primary, m))
    for column in ["effect", "ci_low", "ci_high"]:
        designs[f"std_{column}"] = designs[column] / designs.baseline_sd
    bootstrap = data["rematched"].set_index("estimand").loc[[
        "heart_rate_mean_during_sleep", "heart_rate_mean_during_wake",
        "heart_rate_mean_during_nrem", "heart_rate_mean_during_rem",
        "heart_rate_rem_minus_nrem",
    ]].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.95})
    ax = axes[0]
    offsets = {"primary_optimal_1to1": -0.22, "capped_1to3_reuse2": 0.0,
               "exact_stratum_overlap": 0.22}
    for y, measurement in enumerate(endpoints):
        for row in designs[designs.measurement.eq(measurement)].itertuples(index=False):
            _name, colour, marker = DESIGN_STYLE[row.design]
            interval_point(ax, row.std_effect, row.std_ci_low, row.std_ci_high,
                           y + offsets[row.design], colour=colour, marker=marker,
                           filled=bool(row.gate_pass), size=3.6)
    ax.axvline(0, color=BLACK, linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(endpoints)), [label(m) for m in endpoints])
    ax.set_ylim(len(endpoints) - 0.5, -0.5)
    ax.set_xlabel("DiD (baseline SD)")
    clean_axis(ax)
    ax.legend(handles=[
        *[Line2D([], [], marker=DESIGN_STYLE[d][2], color=DESIGN_STYLE[d][1],
                 linestyle="none", label=DESIGN_STYLE[d][0]) for d in offsets],
        Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=GREY,
               linestyle="none", label="Open: design check failed"),
    ], frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=1)
    panel_label(ax, "a", x=-0.52)

    ax = axes[1]
    for y, row in enumerate(bootstrap.itertuples(index=False)):
        interval_point(ax, row.observed_estimate, row.fixed_match_ci_low,
                       row.fixed_match_ci_high, y - 0.13, colour=VERMILLION, marker="o",
                       filled=True)
        interval_point(ax, row.observed_estimate, row.bootstrap_ci_low,
                       row.bootstrap_ci_high, y + 0.13, colour=GREEN, marker="s", filled=True)
    ax.axvline(0, color=BLACK, linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(bootstrap)), [
        f"{label(m)} (n={int(n)})" for m, n in
        zip(bootstrap.estimand, bootstrap.observed_n_pairs, strict=True)])
    ax.set_ylim(len(bootstrap) - 0.5, -0.5)
    ax.set_xlabel("DiD (bpm)")
    clean_axis(ax)
    ax.legend(handles=[
        Line2D([], [], marker="o", color=VERMILLION, linestyle="none",
               label="Fixed-match bootstrap"),
        Line2D([], [], marker="s", color=GREEN, linestyle="none",
               label=f"Redrawn-match bootstrap ({int(bootstrap.bootstrap_replicates_estimable.min())}"
                     " replicates)"),
    ], frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.17), ncol=1)
    panel_label(ax, "b", x=-0.62)
    designs.to_csv(TABLES / "figure_source_unified_figS4_designs.csv", index=False)
    bootstrap.to_csv(TABLES / "figure_source_unified_figS4_rematched.csv", index=False)
    save(fig, "figS4_alternative_designs")


def figure3_scale(data: dict[str, pd.DataFrame]) -> None:
    per_pair = data["scale_per_pair"]
    contrast = data["primary"].set_index("measurement").loc["heart_rate_rem_minus_nrem"]
    ratio = data["scale_contrasts"].set_index("contrast").loc[
        "REM minus pooled non-REM relative heart-rate DiD"]
    dependence = data["baseline_dependence"].set_index("measurement").loc[[
        "heart_rate_mean_during_sleep", "heart_rate_mean_during_rem",
        "heart_rate_mean_during_nrem", "heart_rate_mean_during_wake",
        "heart_rate_rem_minus_nrem",
    ]].reset_index()
    systemic = data["systemic"].set_index(["scale", "regressor"]).loc[("absolute_bpm", "wake")]
    jitter = np.random.default_rng(20260929).uniform(-0.28, 0.28, len(per_pair))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4),
                             gridspec_kw={"hspace": 0.62, "wspace": 0.55})
    for ax, values, estimate, low, high, null, xlabel, letter in [
        (axes[0, 0], per_pair.rem_minus_nrem_did, contrast.effect, contrast.ci_low,
         contrast.ci_high, 0.0, "REM − non-REM DiD (bpm)", "a"),
        (axes[0, 1], per_pair.ratio_of_ratios, ratio.excess_rise_ratio,
         ratio.excess_rise_ratio_ci_low, ratio.excess_rise_ratio_ci_high, 1.0,
         "Ratio of proportional change,\nREM / non-REM", "b"),
    ]:
        ax.scatter(values, jitter, s=7, color=GREY, alpha=0.55, linewidths=0, zorder=2)
        interval_point(ax, estimate, low, high, 0.62, colour=VERMILLION, marker="D",
                       filled=True, size=5)
        ax.axvline(null, color=BLACK, linestyle="--", linewidth=0.8)
        ax.set_ylim(-0.5, 0.85)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(xlabel)
        clean_axis(ax)
        panel_label(ax, letter, x=-0.08)
    below = int((per_pair.rem_minus_nrem_did < 0).sum())
    axes[0, 0].text(0.98, 0.97, f"{below} of {len(per_pair)} pairs below zero",
                    transform=axes[0, 0].transAxes, ha="right", va="top", fontsize=6.4,
                    color=GREY)

    ax = axes[1, 0]
    y = np.arange(len(dependence))
    for i, row in enumerate(dependence.itertuples(index=False)):
        interval_point(ax, row.oldham_slope, row.oldham_slope_ci_low,
                       row.oldham_slope_ci_high, i, colour=VERMILLION, marker="o",
                       filled=True, size=4)
    ax.axvline(0, color=BLACK, linestyle="--", linewidth=0.8)
    ax.set_yticks(y, [label(m) for m in dependence.measurement])
    ax.set_ylim(len(dependence) - 0.5, -0.5)
    ax.set_xlabel("Change in DiD per bpm of heart-rate level")
    clean_axis(ax)
    panel_label(ax, "c", x=-0.52)

    ax = axes[1, 1]
    ax.scatter(per_pair.wake_did, per_pair.rem_minus_nrem_did, s=8, color=GREY, alpha=0.6,
               linewidths=0, zorder=2)
    grid = np.linspace(systemic.regressor_min, systemic.regressor_max, 50)
    ax.plot(grid, systemic.gap_at_zero_systemic_rise + systemic.amplification_slope * grid,
            color=BLACK, linewidth=1.0, zorder=3)
    interval_point(ax, systemic.gap_at_zero_systemic_rise, systemic.gap_at_zero_ci_low,
                   systemic.gap_at_zero_ci_high, 0.0, colour=VERMILLION, marker="D",
                   filled=True, size=5, horizontal=False)
    ax.axhline(0, color=BLACK, linestyle="--", linewidth=0.8)
    ax.axvline(0, color=LIGHT_GREY, linewidth=0.8, zorder=1)
    ax.set_xlabel("Nocturnal wake DiD, overall rise (bpm)")
    ax.set_ylabel("REM − non-REM DiD (bpm)")
    slope_text = (
        f"slope {systemic.amplification_slope:+.3f} (p={systemic.amplification_slope_p:.2f}); "
        f"excess at zero rise {systemic.gap_at_zero_systemic_rise:+.2f} bpm"
    ).replace("-", "−")
    # Above the axes: every corner of the scatter holds pairs, so an inset label would hide one.
    ax.text(1.0, 1.02, slope_text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.4, color=GREY)
    clean_axis(ax, grid="both")
    panel_label(ax, "d", x=-0.2)
    per_pair.to_csv(TABLES / "figure_source_unified_fig3_per_pair.csv", index=False)
    dependence.to_csv(TABLES / "figure_source_unified_fig3_baseline.csv", index=False)
    save(fig, "fig3_rem_excess_robustness")




def figure_s3_gates(data: dict[str, pd.DataFrame]) -> None:
    status = data["status"].copy()
    status["display"] = status.modality.map(label)
    status = status.sort_values("retention")
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.3), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    y = np.arange(len(status))
    colours = [GREEN if bool(v) else GREY for v in status.design_gate_pass]
    axes[0].scatter(status.retention, y, c=colours, s=24, zorder=3)
    axes[0].axvline(0.85, color=BLACK, linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Retention of eligible initiators")
    axes[0].set_yticks(y, status.display)
    axes[0].set_title("Retention check", loc="left")
    axes[0].set_xlim(0.80, 1.01)
    clean_axis(axes[0])
    panel_label(axes[0], "a", x=-0.62)
    axes[1].scatter(status.max_abs_smd, y, c=colours, s=24, zorder=3)
    axes[1].axvline(0.10, color=BLACK, linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Maximum post-match |SMD|")
    axes[1].set_title("Balance check", loc="left")
    clean_axis(axes[1])
    panel_label(axes[1], "b", x=-0.06)
    axes[0].legend(handles=[
        Line2D([], [], marker="o", color=GREEN, linestyle="none", label="Both checks passed"),
        Line2D([], [], marker="o", color=GREY, linestyle="none", label="At least one check failed"),
    ], frameon=False, loc="upper center", bbox_to_anchor=(1.05, -0.14), ncol=2)
    status.to_csv(TABLES / "figure_source_unified_figS3_design.csv", index=False)
    save(fig, "figS3_design_gates", target_mm=150.0)




def main() -> int:
    configure_style()
    data = load_inputs()
    # Figure 1 is the unified-cohort four-panel phenotyping figure built by
    # `build_fig1_unified.py`; this builder owns Figures 2 onward.
    figure1_sources(data)
    figure2(data)
    figure3_scale(data)
    figure4_hrv(data)
    figure_s1(data)
    figure_s2_arms(data)
    figure_s3_gates(data)
    figure_s4_designs(data)
    figure_s5_exposure(data)
    figure_s6_stage(data)
    print("All unified-cohort paper figures built from versioned aggregate outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
