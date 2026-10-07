"""Write NUMBERS.md and the manuscript tables from the final unified-cohort outputs.

Every number quoted in `manuscript.md` and `supplement.md` should be read from here, so the
text is never typed from memory. The tables in `tables/*.md` are pasted into the manuscript
and supplement verbatim; rerun this script and re-paste when an upstream output changes.

    python paper/build_numbers.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_unified_figures import ALTERNATIVE, FINAL_RUN, LABELS, ROBUST, SECONDARY

PAPER = Path(__file__).resolve().parent
TABLES = PAPER / "tables"
RUNS = FINAL_RUN.parent

UNITS = {
    "sleep:heart_rate": "bpm", "sleep:neurokit_hrv_time_pnn20": "%",
    "sleep:neurokit_hrv_time_rmssd": "ms", "sleep:neurokit_hrv_time_sdnn": "ms",
    "sleep:ahi": "events/h", "sleep:odi": "events/h", "sleep:rdi": "events/h",
    "sleep:sleep_efficiency": "percentage points",
    "blood_pressure:sitting_blood_pressure_pulse_rate": "bpm",
    "blood_pressure:sitting_blood_pressure": "mmHg",
    "dxa:body_comp": "kg", "dxa:total_scan_vat_area": "cm²", "dxa:total_scan_vat_volume": "cm³",
    "anthropometrics:weight": "kg", "anthropometrics:bmi": "kg/m²",
    "anthropometrics:waist": "cm", "anthropometrics:hip": "cm",
    "microbiome:microbiome_clr": "CLR", "microbiome:microbiome_shannon": "index",
}
# DXA masses are stored in grams and reported in kilograms.
GRAMS = ("body_comp_",)
SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def unit(modality: str, measurement: str) -> str:
    for prefix, value in UNITS.items():
        mod, stem = prefix.split(":")
        if modality == mod and measurement.startswith(stem):
            return value
    return ""


def scale(measurement: str) -> float:
    return 1e-3 if measurement.startswith(GRAMS) else 1.0


def num(value: float, digits: int = 2, signed: bool = False) -> str:
    """Unicode minus, optional explicit plus: the manuscript's number style."""
    if value is None or not np.isfinite(value):
        return "NA"
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return text.replace("-", "−")


def est(effect: float, low: float, high: float, digits: int = 2) -> str:
    return f"{num(effect, digits, True)} ({num(low, digits)} to {num(high, digits)})"


def qval(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value >= 0.001:
        return f"{value:.3f}"
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}×10{str(int(exponent)).translate(SUPERSCRIPT)}"


# Manuscript wording rules: sleep stages vs states, "nocturnal wake", no jargon.
TEXT_LABELS = {
    "heart_rate_mean_during_sleep": "Heart rate, whole sleep",
    "heart_rate_mean_during_wake": "Heart rate, nocturnal wake",
    "heart_rate_mean_during_nrem": "Heart rate, non-REM sleep",
    "heart_rate_mean_during_rem": "Heart rate, REM sleep",
    "heart_rate_rem_minus_nrem": "Heart rate, REM minus non-REM sleep",
    "heart_rate_mean_during_light": "Heart rate, light sleep",
    "heart_rate_mean_during_deep": "Heart rate, deep sleep",
    "heart_rate_rem_minus_light": "Heart rate, REM minus light sleep",
    "heart_rate_rem_minus_deep": "Heart rate, REM minus deep sleep",
    "heart_rate_deep_minus_light": "Heart rate, deep minus light sleep",
    "neurokit_hrv_time_pnn20_during_rem": "pNN20, REM sleep",
    "neurokit_hrv_time_rmssd_during_rem": "RMSSD, REM sleep",
    "neurokit_hrv_time_sdnn_during_rem": "SDNN, REM sleep",
    "neurokit_hrv_time_pnn20_during_nrem": "pNN20, non-REM sleep",
    "neurokit_hrv_time_rmssd_during_nrem": "RMSSD, non-REM sleep",
    "neurokit_hrv_time_sdnn_during_nrem": "SDNN, non-REM sleep",
    "ahi": "Apnea-hypopnea index (AHI)", "ahi_4_percent": "AHI, 4% desaturation",
    "ahi_during_rem": "AHI, REM sleep", "odi": "Oxygen desaturation index (ODI)",
    "odi_during_rem": "ODI, REM sleep", "rdi_during_rem": "Respiratory disturbance index, REM sleep",
    "sitting_blood_pressure_pulse_rate": "Sitting pulse",
    **{f"microbiome_clr::{genus}": f"*{genus}*" for genus in (
        "Roseburia", "Desulfovibrio", "Blautia", "Bifidobacterium", "Dorea", "Megasphaera")},
    "microbiome_shannon": "Shannon diversity",
}


def label(measurement: str) -> str:
    return TEXT_LABELS.get(measurement, LABELS.get(measurement, measurement.replace("_", " ")))


def md_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" if i == 0 else "---:" for i in range(len(frame.columns))) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def estimate_rows(frame: pd.DataFrame, measurements: list[str] | None = None) -> pd.DataFrame:
    work = frame if measurements is None else frame.set_index("measurement").loc[
        measurements].reset_index()
    out = []
    for row in work.itertuples(index=False):
        k = scale(row.measurement)
        out.append({
            "Outcome": label(row.measurement), "Unit": unit(row.modality, row.measurement),
            "Pairs": int(row.n_pairs),
            "DiD (95% CI)": est(row.effect * k, row.ci_low * k, row.ci_high * k),
            "q (within modality)": qval(row.fdr_within_modality),
            "Discovery": "yes" if row.claim_eligible else "no",
        })
    return pd.DataFrame(out)


def baseline_table(balance: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = {
        "age": ("Age, years", False), "bmi": ("Body-mass index, kg/m²", False),
        "waist_circumference": ("Waist circumference, cm", False),
        "hip_circumference": ("Hip circumference, cm", False),
        "vat_area": ("Visceral adipose tissue area, cm²", False),
        "gender": ("Female sex", True), "diabetic": ("Diabetes", True),
        "smoking": ("Current smoking", True),
        "on_antihypertensive": ("Antihypertensive therapy", True),
        "on_lipid_lowering": ("Lipid-lowering therapy", True),
        "vat_missing": ("Visceral fat not measured", True),
    }
    out = []
    for covariate, (name, binary) in rows.items():
        row = balance.set_index("covariate").loc[covariate]
        if binary:
            cells = [f"{round(row[f'after_{arm}_mean'] * n)} ({row[f'after_{arm}_mean']:.1%})"
                     for arm in ("treated", "control")]
        else:
            cells = [f"{row[f'after_{arm}_mean']:.1f} ± {row[f'after_{arm}_sd']:.1f}"
                     for arm in ("treated", "control")]
        out.append({"Characteristic": name, f"Semaglutide initiators (n={n})": cells[0],
                    f"Matched controls (n={n})": cells[1], "SMD": num(row.smd_after, 3)})
    return pd.DataFrame(out)


MODALITY_NAMES = {
    "sleep": "Sleep", "blood_pressure": "Blood pressure", "dxa": "Body composition (DXA)",
    "microbiome_function": "Microbiome function", "anthropometrics": "Anthropometrics",
    "microbiome": "Gut taxonomy", "hmo": "Blood tests", "diet": "Diet",
}


# Reading order for every per-modality table: the primary sleep modality, then the other
# modalities that passed their design checks, then those that failed.
MODALITY_ORDER = ["sleep", "blood_pressure", "dxa", "microbiome", "microbiome_function",
                  "anthropometrics", "diet", "hmo"]


def in_modality_order(frame: pd.DataFrame, *also: str) -> pd.DataFrame:
    rank = frame.modality.map({m: i for i, m in enumerate(MODALITY_ORDER)})
    return frame.assign(_rank=rank).sort_values(["_rank", *also], kind="stable").drop(columns="_rank")


def design_table(status: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    discoveries = primary.groupby("modality").claim_eligible.sum()
    out = []
    for row in in_modality_order(status).itertuples(index=False):
        gate = ("not estimable" if row.status == "not_estimable"
                else "pass" if row.design_gate_pass else "fail")
        out.append({
            "Modality": MODALITY_NAMES[row.modality], "Eligible initiators": int(row.eligible_treated),
            "Matched pairs": int(row.matched), "Retention": f"{row.retention:.1%}",
            "Max post-match SMD": f"{row.max_abs_smd:.3f}", "Design checks": gate,
            "Discoveries": int(discoveries.get(row.modality, 0)),
        })
    return pd.DataFrame(out)


def selective_inference_line(primary: pd.DataFrame) -> str:
    """Benjamini-Bogomolov: test inside the R selected of m gated families at q*R/m."""
    gates = primary.dropna(subset=["modality_gate_p"]).drop_duplicates("modality")
    m, r = len(gates), int((gates.modality_gate_fdr < 0.05).sum())
    level = 0.05 * r / m
    claims = primary[primary.claim_eligible]
    survive = claims[claims.fdr_within_modality < level]
    lost = claims[claims.fdr_within_modality >= level]
    return (f"- Selective-inference check: {r} of {m} gated modalities selected, level "
            f"0.05×{r}/{m} = {level:.4f}; {len(survive)} of {len(claims)} discoveries survive; "
            f"not surviving: " + (", ".join(f"{label(x.measurement)} (q={x.fdr_within_modality:.3f})"
                                          for x in lost.itertuples()) or "none"))


def main() -> int:
    TABLES.mkdir(exist_ok=True)
    manifest = json.loads((FINAL_RUN / "run_manifest.json").read_text())
    primary = pd.read_csv(FINAL_RUN / "primary_results.csv")
    status = pd.read_csv(FINAL_RUN / "modality_status.csv")
    balance = pd.read_csv(FINAL_RUN / "balance_sleep.csv")
    timing = pd.read_csv(FINAL_RUN / "timing_diagnostics.csv").set_index("modality")
    robust = {name: pd.read_csv(ROBUST / f"{name}.csv") for name in [
        "design_estimates", "design_diagnostics", "rematched_bootstrap_summary",
        "exposure_definitions", "untreated_drift", "relative_scale_contrasts",
        "baseline_dependence", "systemic_rise_adjustment", "absolute_stage_contrasts",
        "weight_attenuation", "rem_composition", "duration_slopes"]}
    hrv = pd.read_csv(SECONDARY / "hrv_rate_adjustment.csv")
    raw = pd.read_csv(SECONDARY / "raw_stage_results.csv")
    raw_design = pd.read_csv(SECONDARY / "raw_stage_design.csv").iloc[0]
    weighted = pd.read_csv(ALTERNATIVE / "endpoint_estimates.csv")
    weighted_q = pd.read_csv(ALTERNATIVE / "posthoc_selected_hierarchical_results.csv")
    sleep_status = status.set_index("modality").loc["sleep"]
    n_sleep = int(sleep_status.matched)
    # Gate status comes from the run, so a rerun that moves a modality across the gate moves
    # it between these tables too. Microbiome function passes but its 272 pathways, none a
    # discovery, are summarized in the text rather than listed.
    passing = [m for m in status.loc[status.design_gate_pass.astype(bool), "modality"]
               if m != "microbiome_function"]
    failing = list(status.loc[status.status.eq("estimated")
                              & ~status.design_gate_pass.astype(bool), "modality"])

    tables = {
        "table1_baseline": baseline_table(balance, n_sleep),
        "table2_design": design_table(status, primary),
        "table3_primary": estimate_rows(primary[primary.modality.isin(["sleep", "blood_pressure", "dxa"])], [
            "heart_rate_mean_during_sleep", "heart_rate_mean_during_wake",
            "heart_rate_mean_during_nrem", "heart_rate_mean_during_rem",
            "heart_rate_rem_minus_nrem", "ahi", "odi", "sitting_blood_pressure_systolic",
            "sitting_blood_pressure_diastolic", "sitting_blood_pressure_pulse_rate",
            "body_comp_total_fat_mass", "body_comp_total_lean_mass", "total_scan_vat_area"]),
        "tableS1_passing_modalities": estimate_rows(
            in_modality_order(
                primary[primary.modality.isin(passing)]
                .sort_values(["claim_eligible", "fdr_within_modality"], ascending=[False, True]))),
    }
    for name, frame in tables.items():
        (TABLES / f"{name}.md").write_text(md_table(frame) + "\n")

    sections = ["# Locked manuscript numbers",
                "",
                "Generated by `paper/build_numbers.py` from the final unified-cohort outputs. Do not "
                "edit by hand. Quote numbers in the manuscript from here.",
                "",
                "## Cohort",
                "",
                f"- Clean, plausibly dated semaglutide starters: {manifest['dated_clean_initiators']}",
                f"- Participants with any GLP-1 record, excluded from controls: {manifest['never_glp_exclusion_set']}",
                f"- Sleep: {int(sleep_status.pairable)} with a sleep study before and ≥3 months after the start, "
                f"{int(sleep_status.eligible_treated)} eligible, {n_sleep} matched, "
                f"max post-match |SMD| {sleep_status.max_abs_smd:.3f}",
                f"- Claim-eligible outcomes: {manifest['n_claim_eligible']} of {manifest['n_endpoints']} tested",
                selective_inference_line(primary),
                f"- Sleep follow-up: median {timing.loc['sleep', 'treated_followup_median_months']:.1f} months "
                f"(initiators) vs {timing.loc['sleep', 'control_followup_median_months']:.1f} (controls), "
                f"SMD {num(timing.loc['sleep', 'followup_duration_smd'], 3)}",
                f"- Sleep post visit: median {timing.loc['sleep', 'start_to_post_median_months']:.1f} months "
                f"after the recorded start ({int(timing.loc['sleep', 'n_pairs_with_dates'])} matched pairs)",
                "",
                "## Table 1. Baseline characteristics (sleep cohort)", "", md_table(tables["table1_baseline"]), "",
                "## Table 2. Matched design by modality", "", md_table(tables["table2_design"]), "",
                "## All primary estimates in passing modalities ("
                + ", ".join(MODALITY_NAMES[m] for m in passing)
                + "; the 272 microbial pathways are not listed)", "",
                md_table(tables["tableS1_passing_modalities"]), "",
                "## Failed-gate modalities, primary 1:1 (not claim-eligible)", "",
                md_table(estimate_rows(primary[primary.modality.isin(failing)])), ""]

    contrasts = robust["absolute_stage_contrasts"]
    sections += ["## Paired heart-rate contrasts (90 pairs, paired t, BH within the three)", "",
                 md_table(pd.DataFrame([{
                     "Contrast": r.contrast, "n": int(r.n),
                     "Difference (95% CI)": est(r.difference, r.ci_low, r.ci_high),
                     "p": qval(r.p), "q": qval(r.fdr_within_contrast_family)}
                     for r in contrasts.itertuples(index=False)])), ""]

    designs = robust["design_estimates"]
    sections += ["## Alternative designs", "", md_table(pd.DataFrame([{
        "Outcome": label(r.measurement), "Design": r.design, "n treated": int(r.n_treated),
        "Estimate (95% CI)": est(r.effect, r.ci_low, r.ci_high), "Gate": r.gate_pass}
        for r in designs.itertuples(index=False)])), ""]
    sections += ["## Design diagnostics (alternative designs)", "",
                 md_table(robust["design_diagnostics"][["modality", "design", "matched_or_weighted_treated",
                                                        "retention", "max_abs_smd", "design_gate_pass"]].round(3)), ""]

    boot = robust["rematched_bootstrap_summary"]
    sections += ["## Rematched bootstrap", "", md_table(pd.DataFrame([{
        "Outcome": label(r.estimand), "Observed": num(r.observed_estimate, 2, True),
        "Fixed-match 95% CI": f"{num(r.fixed_match_ci_low)} to {num(r.fixed_match_ci_high)}",
        "Rematched 95% CI": f"{num(r.bootstrap_ci_low)} to {num(r.bootstrap_ci_high)}",
        "Rematched mean": num(r.bootstrap_mean), "Replicates": int(r.bootstrap_replicates_estimable)}
        for r in boot.itertuples(index=False)])), ""]

    exposure = robust["exposure_definitions"]
    sections += ["## Exposure definitions", "", md_table(pd.DataFrame([{
        "Exposure": r.exposure, "Exposed": int(r.exposed_participants), "Outcome": label(r.measurement),
        "Design": r.design, "n": int(r.n_treated), "Estimate (95% CI)": est(r.effect, r.ci_low, r.ci_high),
        "Gate": r.gate_pass, "1:1 max |SMD|": f"{r.max_abs_smd_1to1:.3f}"}
        for r in exposure.itertuples(index=False)])), ""]

    ratio = robust["relative_scale_contrasts"]
    sections += ["## Relative scale (ratio of proportional change)", "", md_table(pd.DataFrame([{
        "Contrast": r.contrast, "n": int(r.n),
        "Ratio (95% CI)": f"{r.excess_rise_ratio:.4f} ({r.excess_rise_ratio_ci_low:.4f} to {r.excess_rise_ratio_ci_high:.4f})",
        "p": qval(r.p), "q": qval(r.fdr_within_contrast_family)} for r in ratio.itertuples(index=False)])), ""]
    dep = robust["baseline_dependence"]
    sections += ["## Baseline dependence", "", md_table(pd.DataFrame([{
        "Outcome": label(r.measurement), "n": int(r.n_pairs), "Naive slope": num(r.naive_baseline_slope, 3),
        "Coupling null": num(r.coupling_null_slope, 3),
        "Oldham slope (95% CI)": est(r.oldham_slope, r.oldham_slope_ci_low, r.oldham_slope_ci_high, 3),
        "Oldham p": f"{r.oldham_slope_p:.2f}", "post/pre SD": f"{r.post_pre_sd_ratio:.2f}"}
        for r in dep.itertuples(index=False)])), ""]
    sysr = robust["systemic_rise_adjustment"]
    sections += ["## REM − non-REM gap vs systemic rise", "", md_table(pd.DataFrame([{
        "Scale": r.scale, "Regressor": r.regressor, "n": int(r.n),
        "Slope (95% CI)": est(r.amplification_slope, r.amplification_slope_ci_low, r.amplification_slope_ci_high, 3),
        "Slope p": f"{r.amplification_slope_p:.2f}", "r": num(r.correlation_gap_regressor, 2),
        "Gap at zero (95% CI)": est(r.gap_at_zero_systemic_rise, r.gap_at_zero_ci_low, r.gap_at_zero_ci_high, 3),
        "Regressor range": f"{num(r.regressor_min, 1)} to {num(r.regressor_max, 1)}"}
        for r in sysr.itertuples(index=False)])), ""]

    drift = robust["untreated_drift"]
    sections += ["## Matched controls vs untreated cohort", "", md_table(pd.DataFrame([{
        "Outcome": label(r.measurement), "Untreated n": int(r.n_participants),
        "Interval, years": f"{r.interval_years:.1f}",
        "Untreated change (95% CI)": est(r.drift * scale(r.measurement), r.drift_ci_low * scale(r.measurement),
                                          r.drift_ci_high * scale(r.measurement)),
        "Matched-control change (95% CI)": est(r.control_delta * scale(r.measurement),
                                               r.control_delta_ci_low * scale(r.measurement),
                                               r.control_delta_ci_high * scale(r.measurement))}
        for r in drift.itertuples(index=False)])), ""]

    sections += ["## HRV adjusted for simultaneous heart-rate change", "", md_table(pd.DataFrame([{
        "Model": r.model.split(" adjusted")[0].replace("sleep:", ""), "n": int(r.n),
        "Unadjusted DiD": num(r.unadjusted_pair_did, 2, True),
        "Adjusted intercept (95% CI)": est(r.covariate_adjusted_intercept, r.intercept_ci_low, r.intercept_ci_high),
        "p": f"{r.intercept_p:.2f}"} for r in hrv.itertuples(index=False)])), ""]

    sections += [f"## Raw-channel stage resolution (matched {int(raw_design.matched)} of "
                 f"{int(raw_design.eligible_treated)}, max |SMD| {raw_design.max_abs_smd:.3f}, "
                 f"gate {'pass' if raw_design.design_gate_pass else 'fail'})", "",
                 md_table(pd.DataFrame([{
                     "Outcome": label(r.measurement), "Pairs": int(r.n_pairs),
                     "DiD (95% CI)": est(r.effect, r.ci_low, r.ci_high)}
                     for r in raw.itertuples(index=False)])), ""]

    weight = robust["weight_attenuation"]
    sections += ["## Weight attenuation (same sleep pairs)", "", md_table(pd.DataFrame([{
        "Model": r.model, "n": int(r.n), "Unadjusted": num(r.unadjusted_pair_did, 2, True),
        "Weight-adjusted (95% CI)": est(r.covariate_adjusted_intercept, r.intercept_ci_low, r.intercept_ci_high),
        "Weight slope p": f"{r.covariate_slope_p:.2f}"} for r in weight.itertuples(index=False)])), ""]
    comp = robust["rem_composition"]
    comp = comp[comp.scale.eq("absolute_bpm")]
    sections += ["## Sleep architecture on the same pairs", "", md_table(pd.DataFrame([{
        "Measure": r.composition_measure, "n": int(r.n),
        "DiD (95% CI)": est(r.measure_did, r.measure_did_ci_low, r.measure_did_ci_high),
        "p": f"{r.measure_did_p:.2f}", "Gap slope p": f"{r.amplification_slope_p:.2f}",
        "Gap at zero change (95% CI)": est(r.gap_at_zero_systemic_rise, r.gap_at_zero_ci_low, r.gap_at_zero_ci_high)}
        for r in comp.itertuples(index=False)])), ""]
    dur = robust["duration_slopes"]
    sections += ["## Months on drug", "", md_table(pd.DataFrame([{
        "Outcome": label(r.measurement), "n": int(r.n_pairs),
        "Months, median (range)": f"{r.months_median:.1f} ({r.months_min:.1f} to {r.months_max:.1f})",
        "Slope per month (95% CI)": est(r.slope_per_month, r.slope_ci_low, r.slope_ci_high, 3),
        "p": f"{r.slope_p:.2f}", "q": f"{r.fdr:.2f}"} for r in dur.itertuples(index=False)])), ""]

    sections += ["## Overlap-weighted estimates, failed-gate modalities (Figure 2)", "", md_table(pd.DataFrame([{
        "Modality": MODALITY_NAMES[r.modality], "Outcome": label(r.measurement), "n treated": int(r.n_treated),
        "Estimate (95% CI)": est(r.effect * scale(r.measurement), r.ci_low * scale(r.measurement),
                                 r.ci_high * scale(r.measurement)),
        "Endpoint balance": r.endpoint_balance_pass}
        for r in weighted[weighted.design.eq("exact_stratum_overlap")].itertuples(index=False)])), ""]
    sections += ["## Weighted hierarchical results (post hoc selected designs)", "",
                 md_table(weighted_q.round(4)), ""]

    (PAPER / "NUMBERS.md").write_text("\n".join(sections))
    print(f"wrote {PAPER / 'NUMBERS.md'} and {len(tables)} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
