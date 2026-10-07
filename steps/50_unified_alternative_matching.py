"""Outcome-blind alternative designs for modalities that fail the unified design gate.

This is a sensitivity analysis, not a replacement for the locked primary analysis in
``steps/48_unified_cohort.py``.  It reconstructs exactly the same clean, date-anchored
semaglutide cohort and tests two alternatives for each primary modality whose optimal 1:1
match failed balance in the locked step-48 run (read from its `modality_status.csv`):

1. variable-ratio (up to 1:3) caliper matching, with control reuse capped at two;
2. overlap weighting estimated separately inside the locked exact blocks
   (pre stage, post stage, sex, lipid-lowering therapy).

The design hierarchy is fixed without looking at outcomes: retain the primary design when
it passes; otherwise prefer capped matching if it passes because its treated-population
estimand is closest to the primary analysis; otherwise use exact-stratum overlap weighting
as a sensitivity estimate if it passes.  Effect direction and P values never enter design
selection.  A post-hoc alternative is not silently promoted to the manuscript primary.

Outputs are written to ``outputs/unified_cohort/alternative_matching`` and a concise report
to ``notes/unified_cohort_review_2026-09-22/ALTERNATIVE_MATCHING_REPORT.md``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from scipy import stats


RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "outputs" / "unified_cohort" / "alternative_matching"
REPORT = (
    RUN
    / "notes"
    / "unified_cohort_review_2026-09-22"
    / "ALTERNATIVE_MATCHING_REPORT.md"
)
SOURCE_RUN = (
    RUN
    / "outputs"
    / "unified_cohort"
    / "runs"
    / "semaglutide__floor_3m__stages_main__blocks_plus_lipids__modalities_paper__tabular"
)



def failed_modalities() -> tuple[str, ...]:
    """Primary modalities whose locked 1:1 match was estimated but failed the design gate.

    Read from the source run rather than written down, so a rerun of step 48 that changes a
    modality's gate status cannot leave this analysis testing a stale set. The 2026-10-05
    visit-calendar fix moved gut taxonomy from failing to passing.
    """
    status = pd.read_csv(SOURCE_RUN / "modality_status.csv")
    failed = status[status.status.eq("estimated") & ~status.design_gate_pass.astype(bool)]
    return tuple(failed.modality)


FAILED_MODALITIES = failed_modalities()
FIGURE_CONTEXT_ENDPOINTS = {
    "anthropometrics": (
        "weight",
        "bmi",
        "waist_circumference",
        "hip_circumference",
    ),
    "hmo": (
        "bt_hba1c",
        "bt_ldl_cholesterol",
        "bt_non_hdl_cholesterol",
    ),
    "diet": (
        "energy_kcal_per_day",
        "pct_fat_calories",
        "pct_carb_calories",
        "pct_protein_calories",
    ),
}

DESIGN_PRIMARY = "primary_optimal_1to1"
DESIGN_CAPPED = "capped_1to3_reuse2"
DESIGN_OVERLAP = "exact_stratum_overlap"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def effective_sample_size(weights: pd.Series) -> float:
    values = pd.to_numeric(weights, errors="coerce").dropna().to_numpy(float)
    denominator = np.square(values).sum()
    return float(values.sum() ** 2 / denominator) if denominator > 0 else 0.0


def weighted_moments(values: pd.Series, weights: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(values, errors="coerce").to_numpy(float)
    weights = pd.to_numeric(weights, errors="coerce").to_numpy(float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if not len(values) or weights.sum() <= 0:
        return np.nan, np.nan
    mean = float(np.average(values, weights=weights))
    variance = float(np.average(np.square(values - mean), weights=weights))
    return mean, variance


def weighted_smd(
    treated_values: pd.Series,
    treated_weights: pd.Series,
    control_values: pd.Series,
    control_weights: pd.Series,
) -> float:
    treated_mean, treated_variance = weighted_moments(treated_values, treated_weights)
    control_mean, control_variance = weighted_moments(control_values, control_weights)
    denominator = np.sqrt((treated_variance + control_variance) / 2)
    if not np.isfinite(denominator):
        return np.nan
    return float((treated_mean - control_mean) / denominator) if denominator > 0 else 0.0


def balance_from_pairs(
    step26,
    units: pd.DataFrame,
    pairs: pd.DataFrame,
    modality: str,
    design: str,
) -> pd.DataFrame:
    """Weighted balance; each treated participant has total pair weight one."""
    records = []
    pair_weights = pairs.get("match_weight", pd.Series(1.0, index=pairs.index)).astype(float)
    treated_units = units.loc[units.treated.eq(1)]
    control_units = units.loc[units.treated.eq(0)]
    for covariate in step26.MATCH_COVARIATES:
        before = weighted_smd(
            treated_units[covariate],
            pd.Series(1.0, index=treated_units.index),
            control_units[covariate],
            pd.Series(1.0, index=control_units.index),
        )
        after = weighted_smd(
            pairs[f"treated_{covariate}"],
            pair_weights,
            pairs[f"control_{covariate}"],
            pair_weights,
        )
        records.append(
            {
                "modality": modality,
                "design": design,
                "covariate": covariate,
                "smd_before": before,
                "smd_after": after,
            }
        )
    return pd.DataFrame(records)


def exact_stratum_overlap(
    matching,
    step26,
    units: pd.DataFrame,
    exact_cols: list[str],
) -> pd.DataFrame:
    """Estimate overlap weights independently inside each exact-matching stratum.

    Separate intercepts make the overlap target respect the stage/sex/lipid blocks.  The
    tiny final calibration sets treated and control mass exactly equal inside each stratum;
    with a fitted logistic intercept this is already true up to numerical tolerance.
    """
    frames = []
    grouped = units.groupby(exact_cols, dropna=False, sort=True)
    for block_number, (block_key, block) in enumerate(grouped, start=1):
        if block.treated.nunique() < 2:
            continue
        scored = matching.propensity_score(
            block,
            treatment_col="treated",
            covariate_cols=step26.MATCH_COVARIATES,
            propensity_col="overlap_propensity",
            logit_col="overlap_logit",
        )
        scored["weight"] = np.where(
            scored.treated.eq(1),
            1.0 - scored.overlap_propensity,
            scored.overlap_propensity,
        )
        arm_mass = scored.groupby("treated").weight.sum()
        if 0 not in arm_mass.index or 1 not in arm_mass.index:
            continue
        common_mass = float((arm_mass.loc[0] + arm_mass.loc[1]) / 2)
        for arm in (0, 1):
            scored.loc[scored.treated.eq(arm), "weight"] *= (
                common_mass / float(arm_mass.loc[arm])
            )
        scored["overlap_block"] = block_number
        scored["overlap_block_key"] = repr(block_key)
        frames.append(scored)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def balance_from_weighted_units(
    step26,
    weighted_units: pd.DataFrame,
    modality: str,
    design: str,
) -> pd.DataFrame:
    treated = weighted_units.loc[weighted_units.treated.eq(1)]
    control = weighted_units.loc[weighted_units.treated.eq(0)]
    records = []
    for covariate in step26.MATCH_COVARIATES:
        records.append(
            {
                "modality": modality,
                "design": design,
                "covariate": covariate,
                "smd_before": np.nan,
                "smd_after": weighted_smd(
                    treated[covariate],
                    treated.weight,
                    control[covariate],
                    control.weight,
                ),
            }
        )
    return pd.DataFrame(records)


def block_distribution_difference(
    frame: pd.DataFrame,
    exact_cols: list[str],
    *,
    treated_weight: str,
    control_weight: str | None = None,
) -> float:
    """Largest absolute treated-control difference in exact-stratum proportions."""
    if frame.empty:
        return np.nan
    control_weight = control_weight or treated_weight
    treated = frame.loc[frame.treated.eq(1), [*exact_cols, treated_weight]].copy()
    control = frame.loc[frame.treated.eq(0), [*exact_cols, control_weight]].copy()
    treated = treated.rename(columns={treated_weight: "weight"})
    control = control.rename(columns={control_weight: "weight"})
    treated_mass = treated.groupby(exact_cols, dropna=False).weight.sum()
    control_mass = control.groupby(exact_cols, dropna=False).weight.sum()
    treated_prop = treated_mass / treated_mass.sum()
    control_prop = control_mass / control_mass.sum()
    comparison = pd.concat(
        [treated_prop.rename("treated"), control_prop.rename("control")], axis=1
    ).fillna(0.0)
    return float((comparison.treated - comparison.control).abs().max())


def pair_block_distribution_difference(
    pairs: pd.DataFrame,
    exact_cols: list[str],
) -> float:
    if pairs.empty:
        return np.nan
    weights = pairs.get("match_weight", pd.Series(1.0, index=pairs.index)).astype(float)
    treated = pairs[[*exact_cols]].copy()
    treated["treated"] = 1
    treated["weight"] = weights
    control = pairs[[*exact_cols]].copy()
    control["treated"] = 0
    control["weight"] = weights
    return block_distribution_difference(
        pd.concat([treated, control], ignore_index=True),
        exact_cols,
        treated_weight="weight",
    )


def pair_ess(pairs: pd.DataFrame) -> tuple[float, float]:
    if pairs.empty:
        return 0.0, 0.0
    work = pairs[["treated_id", "control_id"]].copy()
    work["weight"] = pairs.get(
        "match_weight", pd.Series(1.0, index=pairs.index)
    ).astype(float)
    treated_weights = work.groupby("treated_id").weight.sum()
    control_weights = work.groupby("control_id").weight.sum()
    return effective_sample_size(treated_weights), effective_sample_size(control_weights)


def overlap_ess(weighted_units: pd.DataFrame) -> tuple[float, float]:
    participant_weights = (
        weighted_units.groupby(["participant_id", "treated"], as_index=False).weight.sum()
    )
    treated = participant_weights.loc[participant_weights.treated.eq(1), "weight"]
    control = participant_weights.loc[participant_weights.treated.eq(0), "weight"]
    return effective_sample_size(treated), effective_sample_size(control)


def overlap_effects(
    step26,
    weighted_units: pd.DataFrame,
    deltas: pd.DataFrame,
    modality: str,
) -> pd.DataFrame:
    """Weighted DiD with SEs clustered by participant across eligible visit windows."""
    keys = ["participant_id", "pre_stage", "post_stage"]
    covariate_columns = list(step26.MATCH_COVARIATES)
    exact_cols = ["pre_stage", "post_stage", "gender", "on_lipid_lowering"]
    columns = list(dict.fromkeys(
        [*keys, "treated", "weight", "overlap_block", *covariate_columns, *exact_cols]
    ))
    merged = deltas.merge(
        weighted_units[columns].drop_duplicates(keys),
        on=keys,
        how="inner",
    )
    records = []
    for measurement, group in merged.groupby("measurement"):
        group = group.dropna(subset=["delta", "weight"]).copy()
        # Endpoint missingness can disturb the arm equality obtained in the modality-wide
        # overlap population. Recalibrate by observed endpoint availability inside each
        # exact stratum, without using the endpoint value. Strata missing either arm cannot
        # identify a contrast and are excluded for that endpoint.
        calibrated = []
        for _block, stratum in group.groupby("overlap_block", sort=False):
            arm_mass = stratum.groupby("treated").weight.sum()
            if 0 not in arm_mass.index or 1 not in arm_mass.index:
                continue
            common_mass = float((arm_mass.loc[0] + arm_mass.loc[1]) / 2)
            stratum = stratum.copy()
            for arm in (0, 1):
                stratum.loc[stratum.treated.eq(arm), "weight"] *= (
                    common_mass / float(arm_mass.loc[arm])
                )
            calibrated.append(stratum)
        if not calibrated:
            continue
        group = pd.concat(calibrated, ignore_index=True)
        n_treated = group.loc[group.treated.eq(1), "participant_id"].nunique()
        n_controls = group.loc[group.treated.eq(0), "participant_id"].nunique()
        if n_treated < step26.MIN_N or n_controls < step26.MIN_N:
            continue
        endpoint_smds = [
            weighted_smd(
                group.loc[group.treated.eq(1), covariate],
                group.loc[group.treated.eq(1), "weight"],
                group.loc[group.treated.eq(0), covariate],
                group.loc[group.treated.eq(0), "weight"],
            )
            for covariate in covariate_columns
        ]
        participant_weights = (
            group.groupby(["participant_id", "treated"], as_index=False).weight.sum()
        )
        endpoint_ess_t = effective_sample_size(
            participant_weights.loc[participant_weights.treated.eq(1), "weight"]
        )
        endpoint_ess_c = effective_sample_size(
            participant_weights.loc[participant_weights.treated.eq(0), "weight"]
        )
        endpoint_block_difference = block_distribution_difference(
            group, exact_cols, treated_weight="weight"
        )
        endpoint_max_abs_smd = float(np.nanmax(np.abs(endpoint_smds)))
        # Explicit one-way cluster-robust WLS sandwich.  The environment's historical
        # statsmodels build cannot import with the installed compatibility layer, and this
        # two-column model does not need that dependency.  Analytic overlap weights enter
        # both the WLS normal equations and the participant-cluster score contributions.
        y = group.delta.to_numpy(float)
        x = np.column_stack([np.ones(len(group)), group.treated.to_numpy(float)])
        weights = group.weight.to_numpy(float)
        bread = np.linalg.pinv(x.T @ (weights[:, None] * x))
        beta = bread @ (x.T @ (weights * y))
        residual = y - x @ beta
        score_rows = weights[:, None] * x * residual[:, None]
        score_frame = pd.DataFrame(score_rows, columns=["intercept", "treated"])
        score_frame["participant_id"] = group.participant_id.to_numpy()
        cluster_scores = score_frame.groupby("participant_id")[[
            "intercept", "treated"
        ]].sum().to_numpy(float)
        meat = cluster_scores.T @ cluster_scores
        n_clusters = len(cluster_scores)
        n_rows, n_parameters = x.shape
        correction = (
            n_clusters / (n_clusters - 1) * (n_rows - 1) / (n_rows - n_parameters)
            if n_clusters > 1 and n_rows > n_parameters
            else 1.0
        )
        covariance = correction * bread @ meat @ bread
        estimate = float(beta[1])
        standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
        degrees_freedom = max(n_clusters - 1, 1)
        t_statistic = estimate / standard_error if standard_error > 0 else np.nan
        p_value = (
            float(2 * stats.t.sf(abs(t_statistic), degrees_freedom))
            if np.isfinite(t_statistic)
            else np.nan
        )
        critical = float(stats.t.ppf(0.975, degrees_freedom))
        records.append(
            {
                "design": DESIGN_OVERLAP,
                "modality": modality,
                "measurement": measurement,
                "n_treated": n_treated,
                "n_controls": n_controls,
                "effect": estimate,
                "ci_low": estimate - critical * standard_error,
                "ci_high": estimate + critical * standard_error,
                "p": p_value,
                "endpoint_ess_treated": endpoint_ess_t,
                "endpoint_ess_controls": endpoint_ess_c,
                "endpoint_max_abs_smd": endpoint_max_abs_smd,
                "endpoint_exact_block_difference": endpoint_block_difference,
                "endpoint_balance_pass": bool(
                    endpoint_max_abs_smd < 0.10
                    and endpoint_block_difference < 1e-8
                    and min(endpoint_ess_t, endpoint_ess_c) >= step26.MIN_N
                ),
            }
        )
    result = pd.DataFrame(records)
    if not result.empty:
        result["fdr_within_modality"] = step26.bh_fdr(result.p.to_numpy(float))
    return result


def add_pair_endpoint_balance(
    step26,
    pairs: pd.DataFrame,
    deltas: pd.DataFrame,
    effects: pd.DataFrame,
) -> pd.DataFrame:
    """Attach balance of the actual complete links used for each matched endpoint."""
    if effects.empty:
        return effects
    links = step26.pair_effects(pairs, deltas)
    records = []
    for measurement, group in links.groupby("measurement"):
        weights = group.get("match_weight", pd.Series(1.0, index=group.index)).astype(float)
        smds = [
            weighted_smd(
                group[f"treated_{covariate}"],
                weights,
                group[f"control_{covariate}"],
                weights,
            )
            for covariate in step26.MATCH_COVARIATES
        ]
        link_weights = group[["treated_id", "control_id"]].copy()
        link_weights["weight"] = weights.to_numpy()
        treated_weights = link_weights.groupby("treated_id").weight.sum()
        control_weights = link_weights.groupby("control_id").weight.sum()
        maximum = float(np.nanmax(np.abs(smds)))
        records.append(
            {
                "measurement": measurement,
                "endpoint_ess_treated": effective_sample_size(treated_weights),
                "endpoint_ess_controls": effective_sample_size(control_weights),
                "endpoint_max_abs_smd": maximum,
                "endpoint_exact_block_difference": 0.0,
                "endpoint_balance_pass": bool(
                    maximum < 0.10
                    and min(
                        effective_sample_size(treated_weights),
                        effective_sample_size(control_weights),
                    )
                    >= step26.MIN_N
                ),
            }
        )
    return effects.merge(pd.DataFrame(records), on="measurement", how="left")


def primary_effects(result: pd.DataFrame) -> pd.DataFrame:
    return result.rename(
        columns={"n_pairs": "n_treated", "paired_t_p": "p"}
    )[
        [
            "modality",
            "measurement",
            "n_treated",
            "effect",
            "ci_low",
            "ci_high",
            "p",
            "fdr_within_modality",
        ]
    ].assign(design=DESIGN_PRIMARY, n_controls=lambda x: x.n_treated)


def diagnostic_row(
    *,
    modality: str,
    design: str,
    eligible_treated: int,
    matched_treated: int,
    unique_controls: int,
    ess_treated: float,
    ess_controls: float,
    max_control_reuse: int | float,
    max_abs_smd: float,
    block_difference: float,
    estimand: str,
) -> dict:
    retention = matched_treated / eligible_treated if eligible_treated else 0.0
    gate = bool(
        retention >= 0.85
        and max_abs_smd < 0.10
        and block_difference < 1e-8
        and min(ess_treated, ess_controls) >= 8
    )
    return {
        "modality": modality,
        "design": design,
        "estimand": estimand,
        "eligible_treated": eligible_treated,
        "matched_or_weighted_treated": matched_treated,
        "retention": retention,
        "unique_controls": unique_controls,
        "ess_treated": ess_treated,
        "ess_controls": ess_controls,
        "max_control_reuse": max_control_reuse,
        "max_abs_smd": max_abs_smd,
        "max_exact_block_proportion_difference": block_difference,
        "design_gate_pass": gate,
    }


def choose_alternative(rows: pd.DataFrame) -> str:
    passing = set(rows.loc[rows.design_gate_pass, "design"])
    if DESIGN_PRIMARY in passing:
        return DESIGN_PRIMARY
    if DESIGN_CAPPED in passing:
        return DESIGN_CAPPED
    if DESIGN_OVERLAP in passing:
        return DESIGN_OVERLAP
    return "none"


def fmt_number(value: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    show = frame[columns].copy()
    return show.to_markdown(index=False)


def check_report_claims(
    diagnostics: pd.DataFrame,
    effects: pd.DataFrame,
    selected_hierarchical: pd.DataFrame,
) -> None:
    """Fail if the hand-written "Consequences for the paper message" no longer hold.

    The report prose is written by hand. Each sentence is checked here against the computed
    tables, so a rerun that changes a result stops with the claims that broke instead of
    printing stale conclusions.
    """
    failures: list[str] = []

    def endpoint(modality: str, measurement: str) -> pd.Series:
        rows = effects[
            effects.modality.eq(modality)
            & effects.design.eq(DESIGN_OVERLAP)
            & effects.measurement.eq(measurement)
        ]
        if len(rows) != 1:
            raise AssertionError(f"expected one overlap row for {modality}/{measurement}")
        return rows.iloc[0]

    def crosses_zero(row: pd.Series) -> bool:
        return bool(row.ci_low <= 0 <= row.ci_high)

    def hierarchical_hits(modality: str) -> set[str]:
        block = selected_hierarchical[selected_hierarchical.modality.eq(modality)]
        return set(block.loc[block.posthoc_hierarchical_significant.fillna(False).astype(bool),
                             "measurement"])

    overlap_pass = diagnostics[diagnostics.design.eq(DESIGN_OVERLAP)].set_index(
        "modality").design_gate_pass
    for modality in ("anthropometrics", "diet"):
        if not bool(overlap_pass.get(modality, False)):
            failures.append(f"{modality}: overlap design no longer passes the design gate")

    anthropometric = FIGURE_CONTEXT_ENDPOINTS["anthropometrics"]
    for measurement in anthropometric:
        row = endpoint("anthropometrics", measurement)
        if not (row.effect < 0 and row.ci_high < 0):
            failures.append(f"anthropometrics/{measurement}: no longer a clear reduction")
        if not bool(row.endpoint_balance_pass):
            failures.append(f"anthropometrics/{measurement}: fails endpoint balance")
    if hierarchical_hits("anthropometrics") != set(anthropometric):
        failures.append("anthropometrics: not all four survive the hierarchical family")

    hmo = effects[effects.modality.eq("hmo")]
    if hmo.endpoint_balance_pass.fillna(False).astype(bool).any():
        failures.append("hmo: an endpoint now passes complete-case balance")
    for measurement in ("bt_ldl_cholesterol", "bt_non_hdl_cholesterol"):
        if not crosses_zero(endpoint("hmo", measurement)):
            failures.append(f"hmo/{measurement}: overlap interval no longer crosses zero")
    hba1c = endpoint("hmo", "bt_hba1c")
    if not hba1c.effect < 0 or bool(hba1c.endpoint_balance_pass):
        failures.append("hmo/bt_hba1c: no longer negative and imbalanced")

    diet_survivors = {"energy_kcal_per_day", "pct_fat_calories", "pct_carb_calories"}
    for measurement in ("energy_kcal_per_day", "pct_fat_calories"):
        if not endpoint("diet", measurement).ci_high < 0:
            failures.append(f"diet/{measurement}: no longer a clear reduction")
    if not endpoint("diet", "pct_carb_calories").ci_low > 0:
        failures.append("diet/pct_carb_calories: no longer a clear increase")
    if not crosses_zero(endpoint("diet", "pct_protein_calories")):
        failures.append("diet/pct_protein_calories: overlap interval no longer includes zero")
    if hierarchical_hits("diet") != diet_survivors:
        failures.append(f"diet: survivors are {sorted(hierarchical_hits('diet'))}")

    if failures:
        raise AssertionError(
            "ALTERNATIVE_MATCHING_REPORT.md claims no longer match the results; update the "
            "'Consequences for the paper message' prose:\n  - " + "\n  - ".join(failures)
        )


def write_report(
    diagnostics: pd.DataFrame,
    effects: pd.DataFrame,
    overlap_diagnostics: pd.DataFrame,
) -> None:
    choices = {
        modality: choose_alternative(group)
        for modality, group in diagnostics.groupby("modality", sort=False)
    }
    summary = diagnostics.copy()
    summary["retention"] = summary.retention.map(lambda x: fmt_number(x, 3))
    summary["ess_treated"] = summary.ess_treated.map(lambda x: fmt_number(x, 1))
    summary["ess_controls"] = summary.ess_controls.map(lambda x: fmt_number(x, 1))
    summary["max_abs_smd"] = summary.max_abs_smd.map(lambda x: fmt_number(x, 3))
    summary["block_diff"] = summary.max_exact_block_proportion_difference.map(
        lambda x: fmt_number(x, 4)
    )
    summary["pass"] = summary.design_gate_pass.map({True: "PASS", False: "FAIL"})

    context_rows = []
    for modality, measurements in FIGURE_CONTEXT_ENDPOINTS.items():
        block = effects[
            effects.modality.eq(modality) & effects.measurement.isin(measurements)
        ].copy()
        for row in block.itertuples(index=False):
            context_rows.append(
                {
                    "modality": row.modality,
                    "endpoint": row.measurement,
                    "design": row.design,
                    "n": int(row.n_treated),
                    "effect (95% CI)": (
                        f"{fmt_number(row.effect)} "
                        f"[{fmt_number(row.ci_low)}, {fmt_number(row.ci_high)}]"
                    ),
                    "endpoint max abs SMD": fmt_number(row.endpoint_max_abs_smd),
                    "endpoint balance": (
                        "PASS" if bool(row.endpoint_balance_pass) else "FAIL"
                    ),
                }
            )
    context = pd.DataFrame(context_rows)

    decision_lines = []
    for modality in FAILED_MODALITIES:
        chosen = choices[modality]
        chosen_endpoints = effects[
            effects.modality.eq(modality) & effects.design.eq(chosen)
        ]
        n_endpoints = len(chosen_endpoints)
        n_balanced = int(chosen_endpoints.endpoint_balance_pass.fillna(False).sum())
        if chosen == DESIGN_CAPPED:
            text = (
                "capped matching passes and is the preferred alternative sensitivity "
                "because it preserves the treated-population target most closely; "
                f"{n_balanced}/{n_endpoints} endpoints pass complete-case balance"
            )
        elif chosen == DESIGN_OVERLAP:
            if n_balanced:
                text = (
                    "capped matching still fails; exact-stratum overlap weighting passes "
                    f"and {n_balanced}/{n_endpoints} endpoints pass complete-case balance"
                )
            else:
                text = (
                    "modality-wide overlap balance passes, but 0/"
                    f"{n_endpoints} endpoints pass complete-case balance; no endpoint is "
                    "rescued for inferential reporting"
                )
        elif chosen == DESIGN_PRIMARY:
            text = "the locked primary already passes; no alternative is needed"
        else:
            text = "no tested design passes; keep the modality descriptive only"
        decision_lines.append(f"- **{modality}:** {text}.")

    text = f"""# Alternative matching sensitivity — unified semaglutide cohort

Generated from `steps/50_unified_alternative_matching.py` on
{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}.

## Bottom line

The locked optimal 1:1 design remains the manuscript primary. The alternatives below were
selected and evaluated using only retention, effective sample size, exact-block preservation,
and covariate balance. Outcome estimates and P values were calculated only after those design
decisions; they did not determine which method passed.

{chr(10).join(decision_lines)}

Because these alternatives were introduced after observing that the original balance gate
failed, a passing alternative is a **post-hoc sensitivity analysis**, not permission to relabel
the corresponding endpoint as prespecified primary evidence. If it is added to the paper,
state the estimand change explicitly and keep the original failed design visible.

## Design diagnostics

Gate: at least 85% treated retention, maximum absolute weighted SMD <0.10, exact-block
proportion difference <1e-8, and effective sample size of at least eight in each arm. The
overlap model is fitted separately within pre-stage × post-stage × sex × lipid-lowering
strata; its standard errors are clustered by participant because controls can contribute
eligible windows in more than one stratum.

{markdown_table(summary, ['modality', 'design', 'retention', 'ess_treated', 'ess_controls', 'max_abs_smd', 'block_diff', 'pass'])}

## Overlap-weight stability

The exact-stratum models use bounded overlap weights. Participant weights below are summed
over eligible visit-window strata before the maxima are calculated. Some strata contain
only one treated participant; the modest maximum weights and high effective sample sizes
argue against weight instability, but sparse stratum-specific propensity fits are an
additional reason to keep this analysis as sensitivity evidence rather than promote it to
the primary design.

{markdown_table(overlap_diagnostics, ['modality', 'n_exact_strata', 'min_treated_per_stratum', 'min_controls_per_stratum', 'propensity_min', 'propensity_max', 'max_treated_participant_weight', 'max_control_participant_weight'])}

## Prespecified contextual endpoints

This table shows the endpoints already chosen for Figure 2 context (plus all four
anthropometric endpoints), not endpoints selected after seeing this sensitivity. Complete
results for every endpoint are in `endpoint_estimates.csv`.

{markdown_table(context, ['modality', 'endpoint', 'design', 'n', 'effect (95% CI)', 'endpoint max abs SMD', 'endpoint balance'])}

## Consequences for the paper message

- **Anthropometrics:** the passing overlap sensitivity supports reductions in weight, BMI,
  waist circumference, and hip circumference. All four endpoint-complete samples pass
  balance, and all four survive the post-hoc hierarchical sensitivity family.
- **Blood tests (HMO):** do not present HbA1c, LDL, or non-HDL as supported effects. No HMO
  endpoint passes complete-case balance. Under overlap weighting, LDL and non-HDL attenuate
  substantially and their confidence intervals cross zero; HbA1c remains negative but is
  still imbalanced and therefore descriptive only.
- **Diet:** balance is repaired. Energy intake and the fat share of calories fall and the
  carbohydrate share rises; all three survive the post-hoc hierarchical sensitivity family,
  and the protein share does not change. Label these as sensitivity findings, not primary
  discoveries. (Before the 2026-10-05 visit-calendar fix, with 91 rather than 115
  initiators, all four intervals included zero; the carbohydrate share survives since the
  2026-10-06 switch to standard step-up Benjamini-Hochberg.)

## Interpretation rules for the manuscript

- Do not replace the primary 1:1 estimate simply because an alternative is more favorable.
- A capped-matching result targets retained treated initiators and is the closest sensitivity
  to the primary matched-cohort estimand.
- An overlap-weighted result targets participants with clinical equipoise/overlap and is not
  numerically interchangeable with an ATT-like matched estimate.
- Confidence intervals differ by design: paired bootstrap/t-based inference for the primary,
  HC3 weighted regression for capped matching, and participant-clustered weighted regression
  for exact-stratum overlap weighting.
- Endpoint-level balance is recomputed in the actual complete-case sample. For overlap
  weighting, arm mass is re-calibrated within exact strata using endpoint availability only;
  endpoints whose complete-case max absolute SMD remains at least 0.10 stay descriptive.
- Multiplicity in `endpoint_estimates.csv` is within modality and design. These post-hoc
  analyses are not entered into the manuscript's primary hierarchical discovery family.

## Reproducibility

- Cohort: clean single-agent semaglutide initiators, dated start, three-month exposure floor.
- Controls: never reported any GLP-1/GIP–GLP-1 agent.
- Visits: main stages only; last visit before start and first visit at least three months after.
- Exact blocks: pre stage, post stage, sex, and baseline lipid-lowering therapy.
- Matching caliper: 0.2 SD of the propensity-score logit.
- Capped design: up to three controls per treated participant, maximum control reuse of two.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    step48 = load_module("alternative_step48", RUN / "steps" / "48_unified_cohort.py")
    step26 = load_module("alternative_step26", RUN / "steps" / "26_paper_reanalysis.py")
    step31 = load_module("alternative_step31", RUN / "steps" / "31_raw_sleep_glp_cohort.py")
    step47 = load_module("alternative_step47", RUN / "steps" / "47_date_anchored_pairing.py")
    matching = importlib.import_module("research_utils.analysis.models")

    step26.SUPPORTED_STAGES = tuple(step48.MAIN_STAGES)
    exposure = step48.build_exposure(step31, step48.EXPOSURE_SETS["semaglutide"])
    exposure["participant_id"] = step48.as_registration_code(
        exposure.participant, step26
    ).to_numpy()
    covariates = step26.load_covariates()
    all_glp = step48.load_all_glp_reports()
    excluded = set(
        step48.as_registration_code(
            pd.Series(sorted(all_glp.participant.unique())), step26
        )
    )
    canonical_dates = step48.canonical_visit_dates(step26)
    exact_cols = step48.EXACT_BLOCKS[step48.DEFAULT_BLOCKING]
    source_status = pd.read_csv(SOURCE_RUN / "modality_status.csv").set_index("modality")

    diagnostics_records: list[dict] = []
    overlap_diagnostic_records: list[dict] = []
    balances: list[pd.DataFrame] = []
    effects: list[pd.DataFrame] = []

    for modality in FAILED_MODALITIES:
        print(f"--- {modality} ---", flush=True)
        values = step48.load_panel(step26, modality, exposure)
        visit_dates = step48.dates_for(step26, modality, canonical_dates)
        outcome, _windows, _audit = step48.run_instrument(
            step26,
            step31,
            step47,
            modality,
            values,
            exposure,
            visit_dates,
            covariates,
            excluded,
            step48.FLOOR_MONTHS,
            step48.DEFAULT_BLOCKING,
            stages=step48.MAIN_STAGES,
        )
        if not outcome or outcome["result"].empty:
            raise RuntimeError(f"{modality}: primary reconstruction produced no estimate")

        units = outcome["units"]
        primary_pairs = outcome["pairs"]
        deltas = step26.unit_deltas(values, units)
        eligible_treated = int(units.loc[units.treated.eq(1), "participant_id"].nunique())

        # Tripwires against accidentally analysing a different cohort from the final run.
        expected = source_status.loc[modality]
        observed_matched = int(primary_pairs.treated_id.nunique())
        observed_smd = float(outcome["balance"].smd_after.abs().max())
        if observed_matched != int(expected.matched):
            raise AssertionError(
                f"{modality}: reconstructed {observed_matched} primary matches, "
                f"expected {int(expected.matched)}"
            )
        if not np.isclose(observed_smd, float(expected.max_abs_smd), atol=5e-4):
            raise AssertionError(
                f"{modality}: reconstructed max SMD {observed_smd:.6f}, "
                f"expected {float(expected.max_abs_smd):.6f}"
            )

        primary_balance = balance_from_pairs(
            step26, units, primary_pairs, modality, DESIGN_PRIMARY
        )
        balances.append(primary_balance)
        primary_ess_t, primary_ess_c = pair_ess(primary_pairs)
        diagnostics_records.append(
            diagnostic_row(
                modality=modality,
                design=DESIGN_PRIMARY,
                eligible_treated=eligible_treated,
                matched_treated=observed_matched,
                unique_controls=int(primary_pairs.control_id.nunique()),
                ess_treated=primary_ess_t,
                ess_controls=primary_ess_c,
                max_control_reuse=int(primary_pairs.groupby("control_id").size().max()),
                max_abs_smd=float(primary_balance.smd_after.abs().max()),
                block_difference=pair_block_distribution_difference(
                    primary_pairs, exact_cols
                ),
                estimand="retained treated initiators (ATT-like)",
            )
        )
        effects.append(
            add_pair_endpoint_balance(
                step26,
                primary_pairs,
                deltas,
                primary_effects(outcome["result"]),
            )
        )

        capped_pairs = matching.capacity_caliper_match(
            units,
            covariate_cols=step26.MATCH_COVARIATES,
            treatment_col="treated",
            id_col="participant_id",
            exact_cols=exact_cols,
            max_ratio=3,
            control_capacity=2,
            caliper_sd=step48.CALIPER_SD,
            seed=step48.SEED,
        )
        capped_balance = balance_from_pairs(
            step26, units, capped_pairs, modality, DESIGN_CAPPED
        )
        balances.append(capped_balance)
        capped_ess_t, capped_ess_c = pair_ess(capped_pairs)
        diagnostics_records.append(
            diagnostic_row(
                modality=modality,
                design=DESIGN_CAPPED,
                eligible_treated=eligible_treated,
                matched_treated=int(capped_pairs.treated_id.nunique()),
                unique_controls=int(capped_pairs.control_id.nunique()),
                ess_treated=capped_ess_t,
                ess_controls=capped_ess_c,
                max_control_reuse=int(capped_pairs.groupby("control_id").size().max()),
                max_abs_smd=float(capped_balance.smd_after.abs().max()),
                block_difference=pair_block_distribution_difference(capped_pairs, exact_cols),
                estimand="retained treated initiators (ATT-like)",
            )
        )
        capped_effect = step26.weighted_effect(
            capped_pairs,
            deltas,
            design=DESIGN_CAPPED,
            modality=modality,
        )
        effects.append(
            add_pair_endpoint_balance(step26, capped_pairs, deltas, capped_effect)
        )

        overlap_units = exact_stratum_overlap(matching, step26, units, exact_cols)
        overlap_balance = balance_from_weighted_units(
            step26, overlap_units, modality, DESIGN_OVERLAP
        )
        balances.append(overlap_balance)
        overlap_ess_t, overlap_ess_c = overlap_ess(overlap_units)
        participant_weights = (
            overlap_units.groupby(["participant_id", "treated"], as_index=False).weight.sum()
        )
        stratum_counts = (
            overlap_units.groupby(["overlap_block", "treated"])
            .participant_id.nunique()
            .unstack("treated", fill_value=0)
        )
        overlap_diagnostic_records.append(
            {
                "modality": modality,
                "n_exact_strata": int(overlap_units.overlap_block.nunique()),
                "min_treated_per_stratum": int(stratum_counts[1].min()),
                "min_controls_per_stratum": int(stratum_counts[0].min()),
                "propensity_min": round(float(overlap_units.overlap_propensity.min()), 4),
                "propensity_max": round(float(overlap_units.overlap_propensity.max()), 4),
                "max_treated_participant_weight": round(
                    float(
                        participant_weights.loc[
                            participant_weights.treated.eq(1), "weight"
                        ].max()
                    ),
                    4,
                ),
                "max_control_participant_weight": round(
                    float(
                        participant_weights.loc[
                            participant_weights.treated.eq(0), "weight"
                        ].max()
                    ),
                    4,
                ),
            }
        )
        overlap_treated = int(
            participant_weights.loc[
                participant_weights.treated.eq(1), "participant_id"
            ].nunique()
        )
        overlap_controls = int(
            participant_weights.loc[
                participant_weights.treated.eq(0), "participant_id"
            ].nunique()
        )
        diagnostics_records.append(
            diagnostic_row(
                modality=modality,
                design=DESIGN_OVERLAP,
                eligible_treated=eligible_treated,
                matched_treated=overlap_treated,
                unique_controls=overlap_controls,
                ess_treated=overlap_ess_t,
                ess_controls=overlap_ess_c,
                max_control_reuse=float("nan"),
                max_abs_smd=float(overlap_balance.smd_after.abs().max()),
                block_difference=block_distribution_difference(
                    overlap_units,
                    exact_cols,
                    treated_weight="weight",
                ),
                estimand="exact-stratum overlap population",
            )
        )
        effects.append(overlap_effects(step26, overlap_units, deltas, modality))

    diagnostics = pd.DataFrame(diagnostics_records)
    overlap_diagnostics = pd.DataFrame(overlap_diagnostic_records)
    balance = pd.concat(balances, ignore_index=True)
    endpoint_effects = pd.concat(effects, ignore_index=True, sort=False)
    gate_lookup = diagnostics.set_index(["modality", "design"]).design_gate_pass
    endpoint_effects["design_gate_pass"] = [
        bool(gate_lookup.loc[(modality, design)])
        for modality, design in endpoint_effects[["modality", "design"]].itertuples(
            index=False, name=None
        )
    ]

    selection_records = []
    selected_frames = []
    for modality, group in diagnostics.groupby("modality", sort=False):
        chosen = choose_alternative(group)
        selected = endpoint_effects[
            endpoint_effects.modality.eq(modality)
            & endpoint_effects.design.eq(chosen)
        ].copy()
        selected_frames.append(selected[selected.endpoint_balance_pass.fillna(False)])
        selection_records.append(
            {
                "modality": modality,
                "outcome_blind_selected_design": chosen,
                "n_estimable_endpoints": len(selected),
                "n_endpoint_balance_pass": int(
                    selected.endpoint_balance_pass.fillna(False).sum()
                ),
            }
        )
    selections = pd.DataFrame(selection_records)
    selected_balanced = pd.concat(selected_frames, ignore_index=True)
    if not selected_balanced.empty:
        selected_hierarchical = step26.add_hierarchical_multiplicity(
            selected_balanced, p_column="p"
        ).rename(
            columns={
                "modality_gate_p": "posthoc_modality_gate_p",
                "modality_gate_fdr": "posthoc_modality_gate_fdr",
                "hierarchical_significant": "posthoc_hierarchical_significant",
            }
        )
    else:
        selected_hierarchical = selected_balanced.copy()

    OUT.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(OUT / "design_diagnostics.csv", index=False)
    balance.to_csv(OUT / "covariate_balance.csv", index=False)
    overlap_diagnostics.to_csv(OUT / "overlap_weight_diagnostics.csv", index=False)
    endpoint_effects.to_csv(OUT / "endpoint_estimates.csv", index=False)
    selections.to_csv(OUT / "design_selection.csv", index=False)
    selected_hierarchical.to_csv(
        OUT / "posthoc_selected_hierarchical_results.csv", index=False
    )

    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "script": str(Path(__file__).relative_to(RUN)),
        "script_sha256": sha256(Path(__file__)),
        "source_run": str(SOURCE_RUN.relative_to(RUN)),
        "source_run_manifest_sha256": sha256(SOURCE_RUN / "run_manifest.json"),
        "failed_modalities_tested": list(FAILED_MODALITIES),
        "exact_blocks": exact_cols,
        "selection_hierarchy": [DESIGN_PRIMARY, DESIGN_CAPPED, DESIGN_OVERLAP],
        "selection_uses_outcomes": False,
        "gate": {
            "minimum_treated_retention": 0.85,
            "maximum_absolute_smd_exclusive": 0.10,
            "maximum_exact_block_proportion_difference_exclusive": 1e-8,
            "minimum_arm_ess": 8,
        },
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    check_report_claims(diagnostics, endpoint_effects, selected_hierarchical)
    write_report(diagnostics, endpoint_effects, overlap_diagnostics)

    print("\nDesign diagnostics")
    print(
        diagnostics[
            [
                "modality",
                "design",
                "retention",
                "ess_treated",
                "ess_controls",
                "max_abs_smd",
                "design_gate_pass",
            ]
        ].to_string(index=False)
    )
    print("\nOutcome-blind design choices")
    print(selections.to_string(index=False))
    print(f"\nwrote {OUT}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
