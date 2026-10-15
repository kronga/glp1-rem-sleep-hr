"""Does weight or fat lost without a GLP-1 drug move sleep heart rate the same way?

Exploratory analysis: semaglutide initiators lose weight and fat while
their REM-sleep heart rate rises. Among participants who never took a GLP-1 drug, do people
who lose comparable weight or fat "naturally" show a fall in sleep heart rate instead?

Reuses, rather than re-implements, the unified cohort of ``steps/48_unified_cohort.py`` via
the ``Pipeline`` of ``steps/51_unified_sleep_robustness.py``:

* the sleep match is rebuilt and checked against the versioned step-48 run;
* the never-GLP-1 pool is the step-48 control pool (no GLP-1 report of any kind) over the
  initiators' visit windows -- the same pool behind Supplementary Figure S2b;
* within-person changes come from step 26 ``unit_deltas`` on the same unit table.

Never-GLP-1 participants contribute ONE window: their earliest consecutive main-visit window
(00->02, 02->04, 04->06) with WatchPAT mean sleep heart rate and weight at both visits.
Initiators keep their own date-anchored window (pre visit before the start, post visit at
least 3 months after it).

1. Within-person change in sleep heart rate (REM, NREM, all sleep, nocturnal wake), weight,
   BMI, DXA total fat mass and visceral fat area.
2. Slopes of each heart-rate change on weight / fat / VAT / BMI change, adjusted for age,
   sex, baseline BMI and the measured visit interval (HC3), never-GLP-1 and initiators.
   Baseline heart rate is deliberately NOT adjusted for (regression to the mean).
3. "Natural losers" (weight loss >=3 kg, >=3 %, >=5 %, DXA fat loss >=2 kg) against
   never-GLP-1 participants with stable weight (+-1 kg) and against initiators.
4. Initiators matched 1:1 to never-GLP-1 participants on the magnitude of weight (or DXA
   fat) loss plus baseline covariates, then compared on the heart-rate change directly.

Outputs (aggregate only, no identifiers): ``outputs/natural_weight_loss_sleep_hr/``

    python steps/52_natural_weight_loss_sleep_hr.py
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "outputs" / "natural_weight_loss_sleep_hr"

HEART_RATE = {
    "heart_rate_mean_during_rem": "rem",
    "heart_rate_mean_during_nrem": "nrem",
    "heart_rate_mean_during_sleep": "sleep",
    "heart_rate_mean_during_wake": "wake",
}
HEART_RATE_COLUMNS = [f"d_hr_{label}" for label in HEART_RATE.values()]
CONSECUTIVE_WINDOWS = {
    ("00_00_visit", "02_00_visit"),
    ("02_00_visit", "04_00_visit"),
    ("04_00_visit", "06_00_visit"),
}
ADJUSTMENT = ["age", "female", "baseline_bmi", "interval_years"]
# Exposure column -> (label, scale): slopes are reported per `scale` units of exposure.
EXPOSURES = {
    "d_weight": ("weight (kg)", 1.0),
    "d_fat_kg": ("DXA total fat mass (kg)", 1.0),
    "d_vat": ("visceral fat area (10 cm2)", 10.0),
    "d_bmi": ("BMI (kg/m2)", 1.0),
}
STABLE_KG = 1.0
LOSER_DEFINITIONS = {
    "weight_loss_ge_3kg": ("Weight loss >= 3 kg", lambda f: f.d_weight <= -3.0),
    "weight_loss_ge_3pct": ("Weight loss >= 3%", lambda f: f.pct_weight <= -3.0),
    "weight_loss_ge_5pct": ("Weight loss >= 5%", lambda f: f.pct_weight <= -5.0),
    "fat_loss_ge_2kg": ("DXA fat loss >= 2 kg", lambda f: f.d_fat_kg <= -2.0),
}
VERMILION = "#D55E00"
N_BOOT = 2000
SEED = 20261015


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- data

def wide_deltas(step26, values: pd.DataFrame, units: pd.DataFrame,
                columns: dict[str, str]) -> pd.DataFrame:
    """Per-unit pre value and change for the named measurements, one row per window."""
    deltas = step26.unit_deltas(values[values.measurement.isin(columns)], units)
    keys = ["participant_id", "pre_stage", "post_stage"]
    change = deltas.pivot_table(index=keys, columns="measurement", values="delta")
    before = deltas.pivot_table(index=keys, columns="measurement", values="pre_value")
    change = change.rename(columns={k: f"d_{v}" for k, v in columns.items()})
    before = before.rename(columns={k: f"pre_{v}" for k, v in columns.items()})
    return change.join(before)


def unit_table(pipe, sleep: dict) -> pd.DataFrame:
    """Every eligible unit (initiator or never-GLP-1 window) with covariates and changes."""
    step26 = pipe.step26
    units = sleep["units"]
    keys = ["participant_id", "pre_stage", "post_stage"]
    heart = wide_deltas(step26, sleep["values"], units,
                        {k: f"hr_{v}" for k, v in HEART_RATE.items()})
    anthropometrics = wide_deltas(step26, pipe.panel("anthropometrics")[0], units,
                                  {"weight": "weight", "bmi": "bmi"})
    dxa = wide_deltas(step26, pipe.panel("dxa")[0], units,
                      {"body_comp_total_fat_mass": "fat", "total_scan_vat_area": "vat"})
    frame = (units[keys + ["treated", "age", "gender", "bmi", "waist_circumference",
                          "diabetic"]]
             .set_index(keys).join(heart).join(anthropometrics).join(dxa).reset_index())
    frame = frame.rename(columns={"gender": "female", "bmi": "baseline_bmi"})
    # DXA mass is stored in grams.
    frame["d_fat_kg"] = frame.pop("d_fat") / 1000.0
    frame["pre_fat_kg"] = frame.pop("pre_fat") / 1000.0
    frame["pct_weight"] = 100.0 * frame.d_weight / frame.pre_weight

    # Measured interval between the two sleep studies, not the nominal visit spacing.
    sleep_dates = pipe.panel("sleep")[1]
    pre = pd.MultiIndex.from_frame(frame[["participant_id", "pre_stage"]])
    post = pd.MultiIndex.from_frame(frame[["participant_id", "post_stage"]])
    elapsed = (pd.Series(sleep_dates.reindex(post).to_numpy())
               - pd.Series(sleep_dates.reindex(pre).to_numpy()))
    frame["interval_years"] = elapsed.dt.days.to_numpy(float) / 365.25
    frame["matched"] = frame.participant_id.isin(set(sleep["pairs"].treated_id))
    return frame


def analysis_sets(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Never-GLP-1: earliest consecutive window with sleep HR and weight. Initiators: own."""
    usable = frame.dropna(subset=["d_hr_sleep", "d_weight", "interval_years"])
    controls = usable[usable.treated.eq(0)]
    consecutive = [w in CONSECUTIVE_WINDOWS
                   for w in zip(controls.pre_stage, controls.post_stage)]
    controls = (controls[consecutive].sort_values(["participant_id", "pre_stage"])
                .drop_duplicates("participant_id").reset_index(drop=True))
    initiators = usable[usable.treated.eq(1)].reset_index(drop=True)
    return controls, initiators


# --------------------------------------------------------------------------- statistics

def adjusted_fit(step26, frame: pd.DataFrame, outcome: str, terms: list[str]) -> dict:
    """OLS of `outcome` on `terms`, HC3 standard errors; returns the first term's estimate."""
    data = frame.dropna(subset=[outcome, *terms])
    design = np.column_stack([np.ones(len(data)), data[terms].to_numpy(float)])
    beta, se, p = step26.fit_hc3(data[outcome].to_numpy(float), design)
    return {"n": len(data), "estimate": float(beta[1]),
            "ci_low": float(beta[1] - 1.96 * se[1]),
            "ci_high": float(beta[1] + 1.96 * se[1]), "p": float(p[1])}


def slope_table(step26, cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for cohort, frame in cohorts.items():
        for exposure, (label, scale) in EXPOSURES.items():
            scaled = frame.assign(exposure=frame[exposure] / scale)
            for outcome in HEART_RATE_COLUMNS:
                fit = adjusted_fit(step26, scaled, outcome, ["exposure", *ADJUSTMENT])
                records.append({
                    "cohort": cohort, "outcome": outcome, "exposure_change": label,
                    "n": fit["n"],
                    "slope_bpm_per_unit_gained": fit["estimate"],
                    "ci_low": fit["ci_low"], "ci_high": fit["ci_high"], "p": fit["p"],
                    # Sign flipped so that a positive number means "losing raises HR".
                    "bpm_per_unit_lost": -fit["estimate"],
                    "lost_ci_low": -fit["ci_high"], "lost_ci_high": -fit["ci_low"],
                    "adjusted_for": "+".join(ADJUSTMENT),
                })
    return pd.DataFrame(records)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan, np.nan
    draws = rng.choice(values, size=(N_BOOT, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def bootstrap_difference(a: np.ndarray, b: np.ndarray,
                         rng: np.random.Generator) -> tuple[float, float, float]:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    draws = (rng.choice(a, size=(N_BOOT, len(a))).mean(axis=1)
             - rng.choice(b, size=(N_BOOT, len(b))).mean(axis=1))
    return (float(a.mean() - b.mean()), float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)))


def group_profiles(groups: dict[str, pd.DataFrame], rng: np.random.Generator) -> pd.DataFrame:
    records = []
    for group, frame in groups.items():
        row = {
            "group": group, "n_participants": len(frame),
            "age_mean": frame.age.mean(), "female_share": frame.female.mean(),
            "baseline_bmi_mean": frame.baseline_bmi.mean(),
            "interval_years_median": frame.interval_years.median(),
            "d_weight_mean": frame.d_weight.mean(), "pct_weight_mean": frame.pct_weight.mean(),
            "n_with_dxa_fat": int(frame.d_fat_kg.notna().sum()),
            "d_fat_kg_mean": frame.d_fat_kg.mean(),
            "n_with_vat": int(frame.d_vat.notna().sum()), "d_vat_mean": frame.d_vat.mean(),
        }
        for outcome in HEART_RATE_COLUMNS:
            mean, low, high = bootstrap_mean(frame[outcome].to_numpy(float), rng)
            row.update({f"n_{outcome}": int(frame[outcome].notna().sum()),
                        f"{outcome}_mean": mean, f"{outcome}_ci_low": low,
                        f"{outcome}_ci_high": high})
        records.append(row)
    return pd.DataFrame(records)


def group_contrasts(step26, groups: dict[str, pd.DataFrame], reference: str,
                    rng: np.random.Generator) -> pd.DataFrame:
    """Each group minus `reference`: raw bootstrap difference and covariate-adjusted."""
    base = groups[reference]
    records = []
    for group, frame in groups.items():
        if group == reference:
            continue
        combined = pd.concat([frame.assign(in_group=1.0), base.assign(in_group=0.0)])
        for outcome in HEART_RATE_COLUMNS:
            raw, low, high = bootstrap_difference(
                frame[outcome].to_numpy(float), base[outcome].to_numpy(float), rng)
            adjusted = adjusted_fit(step26, combined, outcome, ["in_group", *ADJUSTMENT])
            records.append({
                "group": group, "reference": reference, "outcome": outcome,
                "n_group": int(frame[outcome].notna().sum()),
                "n_reference": int(base[outcome].notna().sum()),
                "raw_difference": raw, "raw_ci_low": low, "raw_ci_high": high,
                "adjusted_difference": adjusted["estimate"],
                "adjusted_ci_low": adjusted["ci_low"], "adjusted_ci_high": adjusted["ci_high"],
                "adjusted_p": adjusted["p"], "adjusted_for": "+".join(ADJUSTMENT),
            })
    return pd.DataFrame(records)


def initiator_effect_at_equal_loss(step26, controls: pd.DataFrame,
                                   initiators: pd.DataFrame) -> pd.DataFrame:
    """Initiator-vs-never-GLP-1 difference in HR change holding the loss itself fixed."""
    combined = pd.concat([initiators.assign(initiator=1.0), controls.assign(initiator=0.0)])
    records = []
    for loss in ["d_weight", "d_fat_kg"]:
        for outcome in HEART_RATE_COLUMNS:
            fit = adjusted_fit(step26, combined, outcome, ["initiator", loss, *ADJUSTMENT])
            records.append({"outcome": outcome, "held_fixed": loss, **fit,
                            "n_initiators": int(combined.dropna(
                                subset=[outcome, loss]).initiator.sum()),
                            "adjusted_for": "+".join([loss, *ADJUSTMENT])})
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- matching

def loss_matched_comparison(pipe, frame: pd.DataFrame, loss: str,
                            rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Initiators matched 1:1 to never-GLP-1 windows on the size of the loss + baseline.

    Same matcher as the paper (optimal 1:1, no control reuse, exact on visit window and
    sex, 0.2 SD propensity caliper, Mahalanobis within it), with the loss and the measured
    interval added to the covariates. Every never-GLP-1 window is a candidate, so a control
    who lost weight between 02 and 04 can match an initiator whose window is 02->04.
    """
    matching = pipe.matching
    covariates = ["age", "baseline_bmi", "waist_circumference", "diabetic",
                  "interval_years", loss]
    units = frame.dropna(subset=["d_hr_rem", *covariates]).rename(
        columns={"female": "gender"}).reset_index(drop=True)
    units = matching.propensity_score(units, treatment_col="treated",
                                      covariate_cols=covariates)
    pairs = matching.optimal_pair_match(
        units, covariate_cols=covariates, treatment_col="treated",
        id_col="participant_id", exact_cols=["pre_stage", "post_stage", "gender"],
        caliper_sd=pipe.step48.CALIPER_SD, seed=SEED,
    )
    keys = ["participant_id", "pre_stage", "post_stage"]
    lookup = units.set_index(keys)
    treated = lookup.reindex(pd.MultiIndex.from_frame(
        pairs[["treated_id", "pre_stage", "post_stage"]])).reset_index(drop=True)
    control = lookup.reindex(pd.MultiIndex.from_frame(
        pairs[["control_id", "pre_stage", "post_stage"]])).reset_index(drop=True)

    balance = []
    for column in dict.fromkeys([*covariates, "gender", "pct_weight", "d_weight", "d_fat_kg"]):
        t, c = treated[column].astype(float), control[column].astype(float)
        pooled = np.sqrt((t.var() + c.var()) / 2)
        balance.append({"matched_on": loss, "variable": column,
                        "initiator_mean": t.mean(), "never_glp1_mean": c.mean(),
                        "smd": (t.mean() - c.mean()) / pooled if pooled > 0 else np.nan})
    records = []
    eligible = int(units.treated.sum())
    for outcome in HEART_RATE_COLUMNS:
        difference = (treated[outcome] - control[outcome]).to_numpy(float)
        valid = np.isfinite(difference)
        mean, low, high = bootstrap_mean(difference, rng)
        _, control_low, control_high = bootstrap_mean(control[outcome][valid].to_numpy(float), rng)
        # Own generator, so adding this interval leaves every earlier draw (and number) unchanged.
        _, treated_low, treated_high = bootstrap_mean(
            treated[outcome][valid].to_numpy(float),
            np.random.default_rng([SEED, len(records), 0 if loss == "d_weight" else 1]))
        records.append({
            "matched_on": loss, "outcome": outcome,
            "eligible_initiators": eligible, "n_pairs": int(valid.sum()),
            "initiator_change": float(treated[outcome][valid].mean()),
            "initiator_change_ci_low": treated_low, "initiator_change_ci_high": treated_high,
            "never_glp1_change": float(control[outcome][valid].mean()),
            "never_glp1_change_ci_low": control_low, "never_glp1_change_ci_high": control_high,
            "difference": mean, "ci_low": low, "ci_high": high,
            "initiator_loss_mean": float(treated[loss][valid].mean()),
            "never_glp1_loss_mean": float(control[loss][valid].mean()),
        })
    return pd.DataFrame(records), pd.DataFrame(balance)


# --------------------------------------------------------------------------- figure

def scatter_figure(controls: pd.DataFrame, initiators: pd.DataFrame,
                   slopes: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = controls.dropna(subset=["d_weight", "d_hr_rem"])
    treated = initiators.dropna(subset=["d_weight", "d_hr_rem"])

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.scatter(data.d_weight, data.d_hr_rem, s=2, color="dimgray", alpha=0.25,
               linewidths=0, zorder=1, label=f"Never GLP-1 (n = {len(data):,})")
    ax.scatter(treated.d_weight, treated.d_hr_rem, s=9, color=VERMILION, alpha=0.85,
               linewidths=0, zorder=3, label=f"Semaglutide initiators (n = {len(treated)})")
    for frame, color, z in [(data, "black", 2), (treated, VERMILION, 4)]:
        slope, intercept = np.polyfit(frame.d_weight, frame.d_hr_rem, 1)
        grid = np.linspace(frame.d_weight.quantile(0.01), frame.d_weight.quantile(0.99), 50)
        ax.plot(grid, intercept + slope * grid, color=color, linewidth=1.0, zorder=z)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5, zorder=0)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.5, zorder=0)

    def label(cohort: str) -> str:
        row = slopes[(slopes.cohort == cohort) & (slopes.outcome == "d_hr_rem")
                     & (slopes.exposure_change == "weight (kg)")].iloc[0]
        return (f"{row.bpm_per_unit_lost:+.2f} bpm per kg lost "
                f"({row.lost_ci_low:+.2f} to {row.lost_ci_high:+.2f})")
    ax.text(0.98, 0.98, "Adjusted slopes (95% CI)\n"
                        f"Never GLP-1: {label('never_glp1')}\n"
                        f"Initiators: {label('initiators_all_eligible')}",
            transform=ax.transAxes, va="top", ha="right", fontsize=5)
    ax.set_xlim(min(data.d_weight.quantile(0.002), treated.d_weight.min()) - 1,
                data.d_weight.quantile(0.998) + 1)
    ax.set_ylim(min(data.d_hr_rem.quantile(0.002), treated.d_hr_rem.min()) - 1,
                max(data.d_hr_rem.quantile(0.998), treated.d_hr_rem.max()) + 7)
    ax.set_xlabel("Change in weight between visits (kg)")
    ax.set_ylabel("Change in REM-sleep heart rate (bpm)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, markerscale=2,
              fontsize=5.5)
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=400, facecolor="white",
                    transparent=False, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- main

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    step51 = load_module("natural_step51", RUN / "steps/51_unified_sleep_robustness.py")
    pipe = step51.Pipeline()
    sleep = pipe.rebuild("sleep", "semaglutide", step51.FINAL_RUN)
    rng = np.random.default_rng(SEED)

    frame = unit_table(pipe, sleep)
    controls, initiators = analysis_sets(frame)
    logging.info("never-GLP-1 participants: %d; initiators: %d", len(controls), len(initiators))

    # Reproduce the paper's within-initiator REM change on the matched initiators.
    matched_rem = sleep["result"].set_index("measurement").loc[
        "heart_rate_mean_during_rem", "treated_delta"]
    # The paper's within-initiator change is over pairs where both members have REM HR.
    rem_pairs = sleep["rows"][sleep["rows"].measurement.eq("heart_rate_mean_during_rem")]
    reproduced = frame.set_index(["participant_id", "pre_stage", "post_stage"]).d_hr_rem.reindex(
        pd.MultiIndex.from_frame(rem_pairs[["treated_id", "pre_stage", "post_stage"]])).mean()
    if not np.isclose(reproduced, matched_rem, rtol=0, atol=1e-9):
        raise AssertionError("matched initiators' REM change does not reproduce the paper")
    logging.info("matched initiators' REM change: paper %.2f, here %.2f",
                 matched_rem, reproduced)

    # Losers-only: is the never-GLP-1 slope carried by weight loss, not only by gain?
    cohorts = {"never_glp1": controls,
               "never_glp1_weight_losers_only": controls[controls.d_weight.lt(0)],
               "initiators_all_eligible": initiators}
    slopes = slope_table(pipe.step26, cohorts)
    slopes.to_csv(OUT / "hr_change_slopes.csv", index=False)

    stable = controls[controls.d_weight.abs().le(STABLE_KG)]
    groups = {"never_glp1_all": controls, "never_glp1_stable_weight_pm1kg": stable}
    for key, (_, rule) in LOSER_DEFINITIONS.items():
        groups[f"never_glp1_{key}"] = controls[rule(controls).fillna(False)]
    groups["initiators_all_eligible"] = initiators
    groups["initiators_matched_paper"] = initiators[initiators.matched]
    group_profiles(groups, rng).to_csv(OUT / "group_changes.csv", index=False)

    never_glp1_groups = {k: v for k, v in groups.items() if k.startswith("never_glp1")}
    pd.concat([
        group_contrasts(pipe.step26, groups, "never_glp1_stable_weight_pm1kg", rng),
        group_contrasts(pipe.step26, {**never_glp1_groups, "initiators_all_eligible": initiators},
                        "initiators_all_eligible", rng),
    ]).to_csv(OUT / "group_contrasts.csv", index=False)

    initiator_effect_at_equal_loss(pipe.step26, controls, initiators).to_csv(
        OUT / "initiator_effect_at_equal_loss.csv", index=False)

    # Matching uses every never-GLP-1 window (not only the one-per-person analysis set) so
    # each initiator's window has candidates, but no control is used twice.
    all_windows = frame.dropna(subset=["d_hr_sleep", "d_weight", "interval_years"])
    matched, balance = zip(*[loss_matched_comparison(pipe, all_windows, loss, rng)
                             for loss in ["d_weight", "d_fat_kg"]])
    pd.concat(matched).to_csv(OUT / "loss_matched_comparison.csv", index=False)
    pd.concat(balance).to_csv(OUT / "loss_matched_balance.csv", index=False)

    scatter_figure(controls, initiators, slopes, OUT / "rem_hr_vs_weight_change")
    # The paper's Figure 4a reads these points; they carry changes only, no identifiers.
    pd.concat([controls.assign(cohort="never_glp1"),
               initiators.assign(cohort="initiators_all_eligible")])[
        ["cohort", "d_weight", "d_hr_rem"]].dropna().to_csv(
        OUT / "rem_hr_vs_weight_points.csv", index=False)
    pd.DataFrame([{
        "never_glp1_participants": len(controls),
        "never_glp1_windows": "; ".join(
            f"{a}->{b}: {n}" for (a, b), n in
            controls.groupby(["pre_stage", "post_stage"]).size().items()),
        "initiators_eligible": len(initiators),
        "initiators_matched": int(initiators.matched.sum()),
        "paper_matched_initiator_rem_change": float(matched_rem),
        "reproduced_matched_initiator_rem_change": float(reproduced),
        "never_glp1_interval_years_median": controls.interval_years.median(),
        "initiators_interval_years_median": initiators.interval_years.median(),
        "n_bootstrap": N_BOOT, "seed": SEED,
    }]).to_csv(OUT / "cohort_summary.csv", index=False)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
