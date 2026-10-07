"""Relative-scale (log) sensitivity for the sleep heart-rate DiDs and state contrasts.

The paper-facing reanalysis estimates every effect on the **absolute** scale:
``did = treated_delta - control_delta`` in native units
(``steps/26_paper_reanalysis.py``). For heart rate in bpm that is the
conventional reporting scale and stays primary.

It is not sufficient for the REM-selectivity claim. ``heart_rate_rem_minus_nrem``
is an arithmetic bpm difference, and REM baseline heart rate is physiologically
higher than pooled non-REM. A purely *multiplicative* heart-rate effect with
**zero** state selectivity therefore widens REM minus non-REM on its own, so the
absolute-scale contrast cannot by itself distinguish state selectivity from
proportional scaling of a higher baseline.

``paper/build_stage_resolved_figures.py`` panel (c) already makes this argument
descriptively: it calibrates a proportional null on pooled non-REM and shows REM
sitting above it. That check is a group-mean comparison with no interval and no
test, computed on the raw-channel estimates rather than the primary estimates the
manuscript reports. This step replaces it with the per-participant version.

Estimator, per matched pair ``i`` and endpoint:

    r_i = [log(treated_post) - log(treated_pre)]
        - [log(control_post) - log(control_pre)]

``exp(mean(r)) - 1`` is the ratio-of-ratios, read as a proportional rise. The
state contrast on this scale is

    c_i = r_i(REM) - r_i(non-REM)

which is algebraically the DiD of the within-visit log ratio
``log(HR_rem / HR_nrem)`` -- i.e. exactly the proportional-scaling null test,
per participant, with an exact interval and a p-value.

The identical matched pairs are reused: this module imports
``steps/26_paper_reanalysis.py`` and calls its ``analyse_modality("sleep", ...)``,
so the cohort, propensity model, optimal 1:1 match and seed are unchanged and the
only thing that differs from the published numbers is the scale.
``reproduction_check.csv`` enforces that, and the step aborts if the recomputed
absolute DiDs do not reproduce ``outputs/paper_reanalysis/primary_1to1_results.csv``.

Pre-specified as a sensitivity. The absolute scale remains primary; this is not a
second path to significance.

Run with the LabData Python environment:

    PYTHONPATH=<repo>/src:<run>/src \
      /net/mraid20/export/jasmine/david/anaconda3/bin/python3 \
      steps/37_relative_scale_sleep_sensitivity.py
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUN = Path(__file__).resolve().parents[1]
STEPS = RUN / "steps"
OUT = RUN / "outputs" / "relative_scale_sleep"
PUBLISHED_PRIMARY = RUN / "outputs" / "paper_reanalysis" / "primary_1to1_results.csv"

MODALITY = "sleep"
STATES = ["sleep", "rem", "nrem", "wake"]
STATE_LABEL = {
    "sleep": "whole sleep",
    "rem": "REM",
    "nrem": "pooled non-REM",
    "wake": "in-study wake",
}
# Calibrating state for the proportional null, matching the choice already made
# in paper/build_stage_resolved_figures.py panel (c).
PROPORTIONAL_NULL_REFERENCE = "nrem"
CONTRAST_PAIRS = [("rem", "nrem"), ("rem", "wake"), ("nrem", "wake")]
CONTRAST_FAMILY = "heart_rate_state_specificity"
# The absolute-scale contrast the manuscript reports. Kept for the reproduction
# check and the baseline-dependence diagnostic, but excluded from the log scale
# because a bpm difference is not sign-constrained.
ABSOLUTE_CONTRAST_ENDPOINT = "heart_rate_rem_minus_nrem"
# Sleep endpoints deliberately left on the absolute scale only: HRV time-domain
# measures and event indices can be zero, so a log transform is undefined for
# them rather than merely inconvenient.
LOG_SCALE_EXCLUSIONS = {
    "neurokit_hrv_time_rmssd_during_rem": "RMSSD can be 0 ms",
    "neurokit_hrv_time_rmssd_during_nrem": "RMSSD can be 0 ms",
    "neurokit_hrv_time_sdnn_during_rem": "SDNN can be 0 ms",
    "neurokit_hrv_time_sdnn_during_nrem": "SDNN can be 0 ms",
    "neurokit_hrv_time_pnn20_during_rem": "pNN20 can be 0 percent",
    "neurokit_hrv_time_pnn20_during_nrem": "pNN20 can be 0 percent",
    "ahi": "event index can be 0 events/hour",
    "ahi_4_percent": "event index can be 0 events/hour",
    "ahi_during_rem": "event index can be 0 events/hour",
    "odi": "event index can be 0 events/hour",
    "odi_during_rem": "event index can be 0 events/hour",
    "rdi_during_rem": "event index can be 0 events/hour",
    "heart_rate_rem_minus_nrem": "bpm difference is not sign-constrained",
    "total_sleep_time": "duration, not a rate under proportional scaling",
    "total_rem_sleep_time": "duration, not a rate under proportional scaling",
    "total_wake_time": "duration, not a rate under proportional scaling",
    "sleep_efficiency": "bounded percentage, not a rate under proportional scaling",
}
REPRODUCTION_TOLERANCE = 1e-9
PAIR_VALUE_COLUMNS = ["treated_pre", "treated_post", "control_pre", "control_post"]


def heart_rate_endpoint(state: str) -> str:
    return f"heart_rate_mean_during_{state}"


def setup_logging() -> None:
    """Local logging only.

    Deliberately does not call the step-26 ``setup_logging``, which truncates the
    published ``outputs/paper_reanalysis/reanalysis.log``.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUT / "relative_scale_sleep.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_reanalysis_step() -> object:
    """Import steps/26_paper_reanalysis.py by path.

    Its module name starts with a digit so it is not importable by name, and it
    guards execution behind ``__main__``, so importing it is side-effect free
    apart from the ``sys.path`` and snapshot assertions it performs at import.
    """
    path = STEPS / "26_paper_reanalysis.py"
    spec = importlib.util.spec_from_file_location("paper_reanalysis_step26", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_sleep_pairs(step26) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-derive the published sleep matched pairs, plus composition on the same pairs.

    Returns the heart-rate pair rows and the sleep-architecture pair rows. The
    latter reuse the very same ``units`` and ``primary_pairs``, so the composition
    check runs on exactly the matched set the contrast is estimated from.
    """
    primary, _sensitivity, _tirzepatide = step26.load_exposure_cohorts()
    ever_glp = step26.ever_glp_participants(primary)
    covariates = step26.load_covariates()
    values = step26.load_snapshot_panel(MODALITY)
    logging.info(
        "Sleep panel: %d participants, %d measurements",
        values["participant_id"].nunique(),
        values["measurement"].nunique(),
    )
    result = step26.analyse_modality(MODALITY, values, primary, covariates, ever_glp)
    if not result:
        raise RuntimeError("analyse_modality returned no sleep result")
    logging.info("Sleep diagnostics: %s", result["diagnostics"])

    composition_values = load_sleep_composition(step26)
    composition_deltas = step26.unit_deltas(composition_values, result["units"])
    composition_rows = step26.pair_effects(result["primary_pairs"], composition_deltas)
    logging.info(
        "Sleep composition: %d pair rows across %d measures",
        len(composition_rows),
        composition_rows["measurement"].nunique() if not composition_rows.empty else 0,
    )
    return result["primary_rows"], composition_rows


def absolute_summary(pair_rows: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """Per-measurement mean absolute DiD, using the step-26 finite/MIN_N filter."""
    records = []
    for measurement, group in pair_rows.groupby("measurement"):
        differences = group["did"].to_numpy(float)
        differences = differences[np.isfinite(differences)]
        if len(differences) < min_n:
            continue
        records.append(
            {
                "measurement": measurement,
                "n_pairs": len(differences),
                "absolute_did": differences.mean(),
            }
        )
    return pd.DataFrame(records)


def reproduction_check(pair_rows: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """Prove the re-derived match reproduces the published sleep numbers."""
    published = pd.read_csv(PUBLISHED_PRIMARY)
    published = published[published["modality"].eq(MODALITY)]
    recomputed = absolute_summary(pair_rows, min_n)
    checked = [heart_rate_endpoint(state) for state in STATES] + [
        ABSOLUTE_CONTRAST_ENDPOINT
    ]
    merged = (
        pd.DataFrame({"measurement": checked})
        .merge(recomputed, on="measurement", how="left")
        .merge(
            published[["measurement", "n_pairs", "effect"]].rename(
                columns={
                    "n_pairs": "published_n_pairs",
                    "effect": "published_absolute_did",
                }
            ),
            on="measurement",
            how="left",
        )
    )
    merged["n_pairs_match"] = merged["n_pairs"].eq(merged["published_n_pairs"])
    merged["absolute_did_abs_diff"] = (
        merged["absolute_did"] - merged["published_absolute_did"]
    ).abs()
    merged["match"] = merged["n_pairs_match"] & merged["absolute_did_abs_diff"].le(
        REPRODUCTION_TOLERANCE
    )
    return merged


def log_scale_pair_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    """Per-pair log-scale DiD for the four state heart-rate endpoints."""
    endpoints = [heart_rate_endpoint(state) for state in STATES]
    work = pair_rows[pair_rows["measurement"].isin(endpoints)].copy()
    for column in PAIR_VALUE_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=PAIR_VALUE_COLUMNS)
    nonpositive = work[PAIR_VALUE_COLUMNS].le(0).any(axis=1)
    if nonpositive.any():
        raise ValueError(
            "Non-positive heart rate reached the log scale; step-26 QC bounds "
            f"heart rate to [30, 140] bpm:\n{work.loc[nonpositive, PAIR_VALUE_COLUMNS]}"
        )
    work["treated_log_delta"] = np.log(work["treated_post"]) - np.log(work["treated_pre"])
    work["control_log_delta"] = np.log(work["control_post"]) - np.log(work["control_pre"])
    work["log_did"] = work["treated_log_delta"] - work["control_log_delta"]
    return work


def as_percent(log_value: float) -> float:
    return 100.0 * (np.expm1(log_value))


def relative_estimates(
    log_rows: pd.DataFrame,
    *,
    seed: int,
    n_boot: int,
    min_n: int,
    bh_fdr,
) -> pd.DataFrame:
    """Per-endpoint relative DiD with the step-26 percentile-bootstrap CI method."""
    rng = np.random.default_rng(seed)
    records = []
    for state in STATES:
        measurement = heart_rate_endpoint(state)
        group = log_rows[log_rows["measurement"].eq(measurement)]
        values = group["log_did"].to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values) < min_n:
            logging.warning("%s: %d pairs below MIN_N, skipped", measurement, len(values))
            continue
        bootstrap = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
        mean_log = float(values.mean())
        try:
            wilcoxon_p = float(stats.wilcoxon(values).pvalue)
        except ValueError:
            wilcoxon_p = np.nan
        records.append(
            {
                "state": state,
                "state_label": STATE_LABEL[state],
                "measurement": measurement,
                "n_pairs": len(values),
                "absolute_did_bpm": float(group["did"].mean()),
                "treated_pre_mean_bpm": float(group["treated_pre"].mean()),
                "control_pre_mean_bpm": float(group["control_pre"].mean()),
                "mean_log_did": mean_log,
                "relative_did_pct": as_percent(mean_log),
                "relative_ci_low_pct": as_percent(float(np.quantile(bootstrap, 0.025))),
                "relative_ci_high_pct": as_percent(float(np.quantile(bootstrap, 0.975))),
                "paired_t_p": float(stats.ttest_1samp(values, 0.0).pvalue),
                "wilcoxon_p": wilcoxon_p,
            }
        )
    result = pd.DataFrame(records)
    if not result.empty:
        result["fdr_within_family"] = bh_fdr(result["paired_t_p"].to_numpy(float))
    return add_proportional_null(result)


def add_proportional_null(estimates: pd.DataFrame) -> pd.DataFrame:
    """Descriptive bpm-scale proportional null, for continuity with figure panel (c).

    One multiplicative rise, calibrated on the reference state, applied to each
    state's own baseline. The tested version of this comparison is the log-scale
    contrast in ``relative_scale_contrasts.csv``; these columns exist so the
    published figure's ~0.9 bpm excess can be read off the primary estimates.
    """
    if estimates.empty:
        return estimates
    reference = estimates[estimates["state"].eq(PROPORTIONAL_NULL_REFERENCE)]
    if reference.empty:
        estimates["proportional_null_did_bpm"] = np.nan
        estimates["excess_over_proportional_null_bpm"] = np.nan
        return estimates
    row = reference.iloc[0]
    rate = float(row["absolute_did_bpm"]) / float(row["treated_pre_mean_bpm"])
    out = estimates.copy()
    out["proportional_null_reference_state"] = PROPORTIONAL_NULL_REFERENCE
    out["proportional_null_rate"] = rate
    out["proportional_null_did_bpm"] = rate * out["treated_pre_mean_bpm"]
    out["excess_over_proportional_null_bpm"] = (
        out["absolute_did_bpm"] - out["proportional_null_did_bpm"]
    )
    return out


def relative_contrasts(log_rows: pd.DataFrame, *, contrast_model, bh_fdr) -> pd.DataFrame:
    """State contrasts on the log scale, reusing the step-26 contrast estimator.

    ``contrast_model`` reads a ``did`` column, so the log-scale DiD is presented
    under that name: the paired t-interval and MIN_N gate are then identical to
    the absolute-scale contrasts in ``formal_specificity_contrasts.csv``.
    """
    scaled = log_rows.assign(did=log_rows["log_did"])
    records = []
    for state_a, state_b in CONTRAST_PAIRS:
        label = f"{STATE_LABEL[state_a]} minus {STATE_LABEL[state_b]} relative heart-rate DiD"
        record = contrast_model(
            scaled,
            heart_rate_endpoint(state_a),
            heart_rate_endpoint(state_b),
            label,
        )
        record["state_a"] = state_a
        record["state_b"] = state_b
        record["multiplicity_family"] = CONTRAST_FAMILY
        records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty or "difference" not in frame.columns:
        return frame
    frame = frame.rename(
        columns={
            "effect_a": "mean_log_did_a",
            "effect_b": "mean_log_did_b",
            "difference": "difference_log",
            "ci_low": "difference_log_ci_low",
            "ci_high": "difference_log_ci_high",
        }
    )
    frame["relative_did_a_pct"] = frame["mean_log_did_a"].map(as_percent)
    frame["relative_did_b_pct"] = frame["mean_log_did_b"].map(as_percent)
    # exp(difference) is the ratio of the two states' proportional rises: 1.0 is
    # the proportional-scaling null.
    frame["excess_rise_ratio"] = np.exp(frame["difference_log"])
    frame["excess_rise_ratio_ci_low"] = np.exp(frame["difference_log_ci_low"])
    frame["excess_rise_ratio_ci_high"] = np.exp(frame["difference_log_ci_high"])
    frame["excess_rise_pct"] = frame["difference_log"].map(as_percent)
    frame["excess_rise_pct_ci_low"] = frame["difference_log_ci_low"].map(as_percent)
    frame["excess_rise_pct_ci_high"] = frame["difference_log_ci_high"].map(as_percent)
    frame["fdr_within_contrast_family"] = np.nan
    finite = frame["p"].notna()
    if finite.any():
        frame.loc[frame.index[finite], "fdr_within_contrast_family"] = bh_fdr(
            frame.loc[finite, "p"].to_numpy(float)
        )
    return frame


def baseline_dependence(pair_rows: pd.DataFrame, *, fit_hc3, min_n: int) -> pd.DataFrame:
    """Does the per-pair absolute DiD depend on the participant's own baseline?

    Two slopes, because the naive one is not interpretable. Regressing a change
    score on its own pre value induces a negative slope even when the true effect
    is uniform, because ``cov(post - pre, pre) = rho*sd_post*sd_pre - sd_pre^2``
    is negative whenever ``rho < sd_post/sd_pre``. ``coupling_null_slope`` is that
    induced value, ``rho*sd_post/sd_pre - 1``, so ``naive_baseline_slope`` can be
    read against what pure regression to the mean would already produce.

    ``oldham_slope`` is the decoupled version: the same regression against the
    treated participant's own pre/post **average** instead of the pre value
    (Oldham's method), which removes the shared ``pre`` term. Only this slope
    supports a claim about effect modification by level. The control arm's delta
    carries no ``treated_pre`` term, so it needs no correction.

    Oldham's method is itself unbiased only when the pre and post error variances
    are equal: unequal variances leave a residual slope proportional to
    ``var(e_post) - var(e_pre)``. ``post_pre_sd_ratio`` is reported so that
    residual can be judged rather than assumed away; a ratio below 1 biases
    ``oldham_slope`` negative.
    """
    endpoints = [heart_rate_endpoint(state) for state in STATES] + [
        ABSOLUTE_CONTRAST_ENDPOINT
    ]
    records = []
    for measurement in endpoints:
        group = pair_rows[pair_rows["measurement"].eq(measurement)].dropna(
            subset=["did", "treated_pre", "treated_post"]
        )
        if len(group) < min_n:
            continue
        difference = group["did"].to_numpy(float)
        pre = group["treated_pre"].to_numpy(float)
        post = group["treated_post"].to_numpy(float)
        naive, naive_se, naive_p = fit_hc3(
            difference, np.column_stack([np.ones(len(group)), pre])
        )
        oldham, oldham_se, oldham_p = fit_hc3(
            difference, np.column_stack([np.ones(len(group)), 0.5 * (pre + post)])
        )
        sd_pre = float(pre.std(ddof=1))
        sd_post = float(post.std(ddof=1))
        correlation = float(np.corrcoef(pre, post)[0, 1])
        records.append(
            {
                "measurement": measurement,
                "n_pairs": len(group),
                "treated_pre_mean_bpm": float(pre.mean()),
                "treated_pre_sd_bpm": sd_pre,
                "treated_post_sd_bpm": sd_post,
                "post_pre_sd_ratio": sd_post / sd_pre if sd_pre > 0 else np.nan,
                "pre_post_correlation": correlation,
                "naive_baseline_slope": float(naive[1]),
                "naive_baseline_slope_ci_low": float(naive[1] - 1.96 * naive_se[1]),
                "naive_baseline_slope_ci_high": float(naive[1] + 1.96 * naive_se[1]),
                "naive_baseline_slope_p": float(naive_p[1]),
                "coupling_null_slope": correlation * sd_post / sd_pre - 1.0
                if sd_pre > 0
                else np.nan,
                "oldham_slope": float(oldham[1]),
                "oldham_slope_ci_low": float(oldham[1] - 1.96 * oldham_se[1]),
                "oldham_slope_ci_high": float(oldham[1] + 1.96 * oldham_se[1]),
                "oldham_slope_p": float(oldham_p[1]),
                "did_change_per_10bpm_higher_level_oldham": float(10.0 * oldham[1]),
            }
        )
    return pd.DataFrame(records)


def _gap_frame(log_rows: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Per-treated REM-minus-non-REM gap alongside candidate systemic-rise regressors."""
    wide = log_rows.pivot_table(
        index="treated_id", columns="measurement", values=value_col
    )
    columns = {state: heart_rate_endpoint(state) for state in STATES}
    missing = [
        columns[state] for state in ("rem", "nrem", "wake") if columns[state] not in wide
    ]
    if missing:
        raise ValueError(f"Missing endpoints for the systemic-rise adjustment: {missing}")
    return pd.DataFrame(
        {
            "gap": wide[columns["rem"]] - wide[columns["nrem"]],
            # Wake is scored on epochs disjoint from both REM and non-REM, so it
            # measures the systemic rise without sharing a term with the gap.
            "wake": wide[columns["wake"]],
            "rem_nrem_mean": 0.5 * (wide[columns["rem"]] + wide[columns["nrem"]]),
            # Shares the subtracted term, so its slope is downward-coupled. Kept
            # so the coupling is visible rather than hidden.
            "nrem": wide[columns["nrem"]],
        }
    )


SYSTEMIC_RISE_REGRESSORS = [
    ("wake", "in-study wake DiD (disjoint epochs, no shared term)", "primary"),
    ("rem_nrem_mean", "mean of REM and non-REM DiD (Oldham-decoupled)", "secondary"),
    ("nrem", "non-REM DiD (shares the subtracted term; downward-coupled)", "diagnostic"),
]


def _adjust_gap(
    gap: pd.Series, regressor: pd.Series, *, fit_hc3, min_n: int, labels: dict
) -> dict | None:
    """Regress the REM-minus-non-REM gap on one candidate explanatory quantity.

    The slope is how much of the gap tracks that quantity; the intercept is the gap
    at zero, i.e. what is left when the quantity explains nothing. Shared by the
    systemic-rise and sleep-composition checks so both report the same columns and
    the same interpretation.
    """
    usable = pd.concat([gap.rename("gap"), regressor.rename("x")], axis=1).dropna()
    if len(usable) < min_n:
        return None
    y = usable["gap"].to_numpy(float)
    x = usable["x"].to_numpy(float)
    beta, standard_error, p_value = fit_hc3(y, np.column_stack([np.ones(len(x)), x]))
    return {
        **labels,
        "n": len(usable),
        "unadjusted_gap": float(y.mean()),
        "correlation_gap_regressor": float(np.corrcoef(x, y)[0, 1]),
        "regressor_mean": float(x.mean()),
        "regressor_min": float(x.min()),
        "regressor_max": float(x.max()),
        "zero_within_regressor_range": bool(x.min() < 0.0 < x.max()),
        "amplification_slope": float(beta[1]),
        "amplification_slope_ci_low": float(beta[1] - 1.96 * standard_error[1]),
        "amplification_slope_ci_high": float(beta[1] + 1.96 * standard_error[1]),
        "amplification_slope_p": float(p_value[1]),
        "gap_at_zero_systemic_rise": float(beta[0]),
        "gap_at_zero_ci_low": float(beta[0] - 1.96 * standard_error[0]),
        "gap_at_zero_ci_high": float(beta[0] + 1.96 * standard_error[0]),
        "gap_at_zero_p": float(p_value[0]),
    }


# Sleep-architecture columns the step-26 sleep panel loads only to gate each
# state's heart rate on >= 5 minutes in that state, then drops before melting
# (steps/26_paper_reanalysis.py:543-553). Reloaded here rather than by editing
# step 26, so the published pipeline stays byte-identical.
COMPOSITION_SOURCE = ("sleep", "sleep_hrv")
COMPOSITION_RAW_COLUMNS = ["total_sleep_time", "total_rem_sleep_time", "total_wake_time"]
COMPOSITION_MEASURES = [
    "percent_rem_of_sleep",
    "rem_minutes",
    "nrem_minutes",
    "total_sleep_minutes",
    "wake_minutes",
]
# The REM-share change is the one that could move REM heart rate for compositional
# rather than physiological reasons, so it leads.
COMPOSITION_PRIMARY = "percent_rem_of_sleep"


def load_sleep_composition(step26) -> pd.DataFrame:
    """Long frame of sleep-architecture measures on the step-26 stage spine."""
    frame, metadata = step26.read_snapshot(
        *COMPOSITION_SOURCE, columns=COMPOSITION_RAW_COLUMNS
    )
    work = step26._participant_and_stage(frame, metadata)
    seconds = {
        column: pd.to_numeric(work[column], errors="coerce")
        for column in COMPOSITION_RAW_COLUMNS
    }
    sleep_seconds = seconds["total_sleep_time"]
    rem_seconds = seconds["total_rem_sleep_time"]
    work["total_sleep_minutes"] = sleep_seconds / 60.0
    work["rem_minutes"] = rem_seconds / 60.0
    work["nrem_minutes"] = (sleep_seconds - rem_seconds) / 60.0
    work["wake_minutes"] = seconds["total_wake_time"] / 60.0
    share = 100.0 * rem_seconds.where(sleep_seconds > 0) / sleep_seconds.where(
        sleep_seconds > 0
    )
    work["percent_rem_of_sleep"] = share.where(share.between(0.0, 100.0))
    return step26._to_long(work, COMPOSITION_MEASURES)


def rem_composition_check(
    log_rows: pd.DataFrame,
    composition_rows: pd.DataFrame,
    *,
    fit_hc3,
    min_n: int,
) -> pd.DataFrame:
    """Could the REM excess be compositional rather than physiological?

    REM heart rate is averaged over epochs *labelled* REM, so a treatment-induced
    shift in sleep architecture would change which epochs contribute. Two questions
    per measure, on the same 41 matched pairs the contrast is estimated from:

    1. Did the measure itself move under treatment? (``measure_did``, paired t.)
    2. Does the REM-minus-non-REM gap track each participant's own change in it,
       and what is left when it does not? (slope and intercept, as in
       :func:`systemic_rise_adjustment`.)

    Existing evidence against compositional drift comes from the n=97
    staging-verification slice rather than these pairs, and
    ``composition_check.csv`` tests only *deep*-sleep composition, never REM.
    """
    records = []
    composition = composition_rows.pivot_table(
        index="treated_id", columns="measurement", values="did"
    )
    for scale, value_col in [("absolute_bpm", "did"), ("relative_log", "log_did")]:
        gap = _gap_frame(log_rows, value_col)["gap"]
        for measure in COMPOSITION_MEASURES:
            if measure not in composition:
                logging.warning("Composition measure %s unavailable", measure)
                continue
            record = _adjust_gap(
                gap,
                composition[measure],
                fit_hc3=fit_hc3,
                min_n=min_n,
                labels={
                    "scale": scale,
                    "composition_measure": measure,
                    "role": "primary" if measure == COMPOSITION_PRIMARY else "secondary",
                },
            )
            if record is None:
                continue
            # Did the measure itself change under treatment? Restricted to exactly
            # the pairs the regression above used, so the two are comparable. This
            # is the on-cohort version of the staging check.
            values = (
                pd.concat([gap.rename("gap"), composition[measure].rename("x")], axis=1)
                .dropna()["x"]
                .to_numpy(float)
            )
            interval = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(
                len(values)
            )
            record["measure_did"] = float(values.mean())
            record["measure_did_ci_low"] = float(values.mean() - interval)
            record["measure_did_ci_high"] = float(values.mean() + interval)
            record["measure_did_p"] = float(stats.ttest_1samp(values, 0.0).pvalue)
            records.append(record)
    return pd.DataFrame(records)


def systemic_rise_adjustment(
    log_rows: pd.DataFrame, *, fit_hc3, min_n: int
) -> pd.DataFrame:
    """Is the REM excess a scaled reflection of the systemic GLP-1 heart-rate rise?

    A purely *additive* global rise cancels exactly in ``REM - non-REM``, and a
    purely *proportional* one cancels on the log scale, so neither can by itself
    produce the contrast. The open question is different: whether the REM excess
    is a fixed amplification of *how large* each participant's systemic rise was,
    in which case there is no REM-specific mechanism, only gain.

    That is a slope-versus-intercept question. The slope on a systemic-rise
    measure is the amplification; the **intercept is the REM excess at zero
    systemic rise**. An intercept indistinguishable from zero with a positive
    slope means pure gain. An intercept that survives means a state-specific
    component that scaling does not explain.

    ``regressor_min`` / ``regressor_max`` are reported so the intercept can be
    confirmed as interpolation rather than extrapolation beyond the observed data.
    """
    records = []
    for scale, value_col in [("absolute_bpm", "did"), ("relative_log", "log_did")]:
        frame = _gap_frame(log_rows, value_col)
        for name, description, role in SYSTEMIC_RISE_REGRESSORS:
            record = _adjust_gap(
                frame["gap"],
                frame[name],
                fit_hc3=fit_hc3,
                min_n=min_n,
                labels={
                    "scale": scale,
                    "regressor": name,
                    "regressor_description": description,
                    "role": role,
                },
            )
            if record is not None:
                records.append(record)
    return pd.DataFrame(records)


def verdict(contrasts: pd.DataFrame) -> tuple[str, str]:
    """Apply the pre-specified decision rule to the REM minus non-REM contrast."""
    row = contrasts[contrasts["state_a"].eq("rem") & contrasts["state_b"].eq("nrem")]
    if row.empty or pd.isna(row.iloc[0].get("fdr_within_contrast_family")):
        return "not estimable", "The relative-scale REM contrast could not be estimated."
    q = float(row.iloc[0]["fdr_within_contrast_family"])
    if q < 0.05:
        return (
            "scale-robust",
            f"q={q:.4g} < 0.05. REM selectivity survives on the relative scale, so the "
            "absolute-scale contrast is not explained by proportional scaling of a "
            "higher REM baseline. The published claim stands and should cite this "
            "sensitivity.",
        )
    if q < 0.10:
        return (
            "weakened",
            f"q={q:.4g} is between 0.05 and 0.10. REM selectivity is directionally "
            "preserved but not clearly separable from proportional scaling. The "
            "state-selectivity language needs softening.",
        )
    return (
        "scale-dependent",
        f"q={q:.4g} >= 0.10. On the relative scale REM is not distinguishable from "
        "pooled non-REM, so the absolute-scale contrast is consistent with one "
        "multiplicative heart-rate effect applied to a higher REM baseline. The "
        "contrast must be reframed before submission.",
    )


def format_table(frame: pd.DataFrame, columns: list[str], precision: int = 4) -> str:
    subset = frame[[c for c in columns if c in frame.columns]]
    return subset.to_markdown(index=False, floatfmt=f".{precision}f")


def write_readme(
    reproduction: pd.DataFrame,
    estimates: pd.DataFrame,
    contrasts: pd.DataFrame,
    dependence: pd.DataFrame,
    systemic: pd.DataFrame,
    composition: pd.DataFrame,
) -> None:
    status, explanation = verdict(contrasts)
    reproduced = bool(reproduction["match"].all())
    text = f"""# Relative-scale sensitivity: sleep heart rate

Generated by `steps/37_relative_scale_sleep_sensitivity.py`. Pre-specified
**sensitivity**: the absolute-scale estimates in
`outputs/paper_reanalysis/primary_1to1_results.csv` and
`outputs/paper_reanalysis/formal_specificity_contrasts.csv` remain primary. This
is not a second path to significance.

## Why

`heart_rate_rem_minus_nrem` is an arithmetic bpm difference, and REM baseline
heart rate exceeds pooled non-REM. One multiplicative heart-rate effect with no
state selectivity therefore widens REM minus non-REM on its own. The
absolute-scale contrast cannot separate state selectivity from proportional
scaling; this step does, per participant.

## Design

Identical matched set to the published sleep analysis: this step calls
`analyse_modality("sleep", ...)` from `steps/26_paper_reanalysis.py`, so the
cohort, propensity model, exact blocks, caliper, seed and optimal 1:1 match are
unchanged. Only the outcome scale differs.

Per matched pair and endpoint:

    r_i = [log(treated_post) - log(treated_pre)] - [log(control_post) - log(control_pre)]

`exp(mean(r)) - 1` is the ratio-of-ratios, reported as a percent. The state
contrast `c_i = r_i(a) - r_i(b)` is the DiD of the within-visit log ratio
`log(HR_a / HR_b)`, i.e. the proportional-scaling null test with an exact
interval. CI methods mirror their absolute-scale counterparts exactly:
percentile bootstrap of the mean for per-endpoint estimates, paired t-interval
for contrasts, BH-FDR within the `{CONTRAST_FAMILY}` family.

## Reproduction check

Re-derived absolute DiDs versus the published sleep numbers. All rows must
match, otherwise the relative-scale numbers are not comparable to the published
ones.

**Reproduced: {reproduced}**

{format_table(reproduction, ["measurement", "n_pairs", "published_n_pairs", "absolute_did", "published_absolute_did", "absolute_did_abs_diff", "match"], 6)}

## Relative-scale DiD by state

{format_table(estimates, ["state_label", "n_pairs", "absolute_did_bpm", "treated_pre_mean_bpm", "relative_did_pct", "relative_ci_low_pct", "relative_ci_high_pct", "paired_t_p", "fdr_within_family"])}

Descriptive proportional null, calibrated on {STATE_LABEL[PROPORTIONAL_NULL_REFERENCE]}
and applied to each state's own baseline. This reproduces the comparison drawn
in `paper/build_stage_resolved_figures.py` panel (c) on the primary estimates;
the tested version is the contrast table below.

{format_table(estimates, ["state_label", "absolute_did_bpm", "proportional_null_did_bpm", "excess_over_proportional_null_bpm"], 3)}

## State contrasts on the relative scale

`excess_rise_ratio` is the ratio of the two states' proportional rises. **1.0 is
the proportional-scaling null.**

{format_table(contrasts, ["contrast", "n", "relative_did_a_pct", "relative_did_b_pct", "excess_rise_ratio", "excess_rise_ratio_ci_low", "excess_rise_ratio_ci_high", "p", "fdr_within_contrast_family"])}

### Verdict on REM selectivity: {status}

{explanation}

Pre-specified decision rule, on the relative-scale REM minus non-REM
`fdr_within_contrast_family`:

| q | Reading |
|---|---|
| < 0.05 | scale-robust; claim stands, cite this sensitivity |
| 0.05 - 0.10 | weakened; soften the state-selectivity language |
| >= 0.10 | scale-dependent; the contrast needs reframing |

## Is the REM excess just the systemic rise passing through?

GLP-1 raises heart rate systemically, and both state means are read off the same
overnight signal. Three distinct versions of that concern:

1. A purely **additive** global rise cancels exactly in `REM - non-REM`. Cannot
   produce the contrast, by construction.
2. A purely **proportional** global rise does not cancel in the absolute
   difference but does cancel on the log scale. That is the contrast table above.
3. The REM excess could be a fixed **amplification of how large each
   participant's own systemic rise was** -- pure gain, no REM-specific mechanism.
   Neither 1 nor 2 addresses this. This table does.

Slope versus intercept: the slope is the amplification, and
`gap_at_zero_systemic_rise` is **the REM excess at zero systemic rise**. Pure gain
means an intercept indistinguishable from zero with a positive slope. A surviving
intercept means a state-specific component that scaling does not explain.
`zero_within_regressor_range` confirms the intercept is interpolation.

{format_table(systemic, ["scale", "regressor", "role", "n", "correlation_gap_regressor", "amplification_slope", "amplification_slope_p", "gap_at_zero_systemic_rise", "gap_at_zero_ci_low", "gap_at_zero_ci_high", "gap_at_zero_p", "zero_within_regressor_range"])}

Read the `wake` rows first: in-study wake is scored on epochs disjoint from both
REM and non-REM, so it measures the systemic rise without sharing a term with the
gap. The `nrem` rows share the subtracted term and are downward-coupled by
construction; they are shown so that coupling is visible rather than hidden.

## Is the REM excess compositional rather than physiological?

REM heart rate is averaged over epochs *labelled* REM, so a treatment-induced
shift in sleep architecture would change which epochs contribute. Same 41 matched
pairs as the contrast, so this is the on-cohort version of the staging check --
the existing evidence comes from the n=97 staging-verification slice, and
`composition_check.csv` tests only *deep*-sleep composition, never REM.

Did the architecture itself move under treatment? `percent_rem_of_sleep` is the
measure that could shift REM heart rate for compositional reasons, so it leads.

{format_table(composition[composition["scale"].eq("absolute_bpm")], ["composition_measure", "role", "n", "measure_did", "measure_did_ci_low", "measure_did_ci_high", "measure_did_p"], 3)}

Does the gap track each participant's own architecture change, and what is left
when it does not? `gap_at_zero_systemic_rise` is the gap at zero architecture
change.

{format_table(composition, ["scale", "composition_measure", "n", "correlation_gap_regressor", "amplification_slope", "amplification_slope_p", "gap_at_zero_systemic_rise", "gap_at_zero_ci_low", "gap_at_zero_ci_high", "gap_at_zero_p"])}

## Baseline dependence of the absolute DiD

Does one absolute DiD average genuinely different per-participant effects? Two
slopes, because the naive one is not interpretable on its own.

`naive_baseline_slope` regresses the per-pair DiD on the treated participant's
own pre value. That is mathematically coupled: `cov(post - pre, pre)` is negative
whenever `rho < sd_post / sd_pre`, so a negative slope appears even when the true
effect is uniform. `coupling_null_slope` is the value pure regression to the mean
already produces, `rho * sd_post / sd_pre - 1`. **Compare the two before reading
anything into the naive slope.**

{format_table(dependence, ["measurement", "n_pairs", "pre_post_correlation", "naive_baseline_slope", "naive_baseline_slope_p", "coupling_null_slope"])}

`oldham_slope` is the decoupled version: the same regression against the treated
participant's own pre/post **average** rather than the pre value (Oldham's
method), which removes the shared `pre` term. The control arm's delta carries no
`treated_pre` term, so it needs no correction. **Only this slope supports a claim
about effect modification by heart-rate level.**

{format_table(dependence, ["measurement", "n_pairs", "post_pre_sd_ratio", "oldham_slope", "oldham_slope_ci_low", "oldham_slope_ci_high", "oldham_slope_p", "did_change_per_10bpm_higher_level_oldham"])}

Oldham's method is itself unbiased only when the pre and post error variances are
equal. `post_pre_sd_ratio` below 1 leaves a residual **negative** bias in
`oldham_slope`, so read the slope against that ratio rather than treating it as
exact.

This whole diagnostic is a *within-state, between-participant* question and is
intrinsically vulnerable to the coupling above. It is secondary and descriptive.
The REM-selectivity question is *within-participant, between-state*, so the
log-scale contrast table is immune to all of this and remains the load-bearing
test here.

## Endpoints deliberately left on the absolute scale

A log transform is undefined, not merely awkward, for endpoints that can reach
zero or change sign. These stay absolute-scale only:

{chr(10).join(f"- `{name}` -- {reason}" for name, reason in sorted(LOG_SCALE_EXCLUSIONS.items()))}

## Not covered here

The raw-channel stage-resolved analysis (`steps/32_raw_sleep_matched_did.py`),
which is the only source of light (N1+N2) and deep (N3) heart rate and of the
three exposure arms, does not persist its participant-level pre/post panel, so a
relative-scale version there requires a full raw-channel re-run. Deferred.
"""
    (OUT / "README.md").write_text(text)


def main() -> None:
    setup_logging()
    step26 = load_reanalysis_step()
    logging.info("Reusing step-26 matching with SEED=%d", step26.SEED)

    pair_rows, composition_rows = build_sleep_pairs(step26)

    reproduction = reproduction_check(pair_rows, step26.MIN_N)
    reproduction.to_csv(OUT / "reproduction_check.csv", index=False)
    logging.info("Reproduction check:\n%s", reproduction.to_string(index=False))
    if not reproduction["match"].all():
        raise RuntimeError(
            "Re-derived sleep DiDs do not reproduce "
            f"{PUBLISHED_PRIMARY}; see {OUT / 'reproduction_check.csv'}. "
            "The relative-scale estimates would not be comparable to the "
            "published absolute-scale ones."
        )

    log_rows = log_scale_pair_rows(pair_rows)
    log_rows.to_csv(OUT / "pair_level_log_did.csv", index=False)

    estimates = relative_estimates(
        log_rows,
        seed=step26.SEED,
        n_boot=step26.N_BOOT,
        min_n=step26.MIN_N,
        bh_fdr=step26.bh_fdr,
    )
    estimates.to_csv(OUT / "relative_scale_did.csv", index=False)
    logging.info("Relative-scale DiD:\n%s", estimates.to_string(index=False))

    contrasts = relative_contrasts(
        log_rows,
        contrast_model=step26.contrast_model,
        bh_fdr=step26.bh_fdr,
    )
    contrasts.to_csv(OUT / "relative_scale_contrasts.csv", index=False)
    logging.info("Relative-scale contrasts:\n%s", contrasts.to_string(index=False))

    systemic = systemic_rise_adjustment(
        log_rows, fit_hc3=step26.fit_hc3, min_n=step26.MIN_N
    )
    systemic.to_csv(OUT / "systemic_rise_adjustment.csv", index=False)
    logging.info("Systemic-rise adjustment:\n%s", systemic.to_string(index=False))

    composition = rem_composition_check(
        log_rows, composition_rows, fit_hc3=step26.fit_hc3, min_n=step26.MIN_N
    )
    composition.to_csv(OUT / "rem_composition_check.csv", index=False)
    logging.info("REM composition check:\n%s", composition.to_string(index=False))

    dependence = baseline_dependence(
        pair_rows, fit_hc3=step26.fit_hc3, min_n=step26.MIN_N
    )
    dependence.to_csv(OUT / "baseline_dependence.csv", index=False)
    logging.info("Baseline dependence:\n%s", dependence.to_string(index=False))

    write_readme(reproduction, estimates, contrasts, dependence, systemic, composition)
    status, explanation = verdict(contrasts)
    logging.info("REM selectivity on the relative scale: %s -- %s", status, explanation)
    logging.info("Relative-scale sleep sensitivity complete: %s", OUT)


if __name__ == "__main__":
    main()
