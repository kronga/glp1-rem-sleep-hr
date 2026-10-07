"""Sleep-robustness analyses on the final unified (date-anchored) semaglutide cohort.

The visit-transition paper carried four robustness analyses that were never ported when the
paper moved to the unified cohort of ``steps/48_unified_cohort.py``. This step reruns them on
that cohort by reusing the original estimators rather than re-implementing them:

1. Alternative designs -- the primary no-reuse 1:1 match, capped control reuse (1:3, each
   control at most twice) and exact-stratum overlap weighting, estimators from step 50.
2. Rematched bootstrap -- resample initiators, refit the propensity model and redraw the
   match inside every replicate (step 29's resampler and summary), so the interval carries
   match-selection uncertainty that the fixed-match bootstrap conditions away.
3. Untreated-cohort drift -- the matched controls' change against every eligible
   never-GLP-1 participant over the same visit windows (step 26 ``untreated_drift``).
4. Relative scale -- ratio of proportional heart-rate changes, naive and Oldham baseline
   slopes, and the REM-minus-non-REM gap against the wake rise (step 37 estimators).
5. Exposure definitions -- the same endpoints for any-GLP-1 and non-semaglutide GLP-1
   starters, 1:1 and overlap weighted, beside the primary semaglutide cohort.
6. Interpretation checks on the same sleep pairs -- REM heart rate adjusted for the pair's
   weight change (step 26 ``attenuation_model``), the change in REM share of sleep and
   whether the REM-minus-non-REM gap tracks it (step 37 composition check), and the slope of
   each DiD on months on drug at the post visit.

Every reconstructed match is checked against a versioned step-48 run before anything is
fitted. Participant identifiers and matched-pair rows stay in memory; outputs are aggregate
estimates plus one id-free, value-sorted per-pair table for the relative-scale figure.

Outputs: ``outputs/unified_cohort/sleep_robustness/``

    python steps/51_unified_sleep_robustness.py [--n-bootstrap 500]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
RUNS = RUN / "outputs/unified_cohort/runs"
RUN_SUFFIX = "floor_3m__stages_main__blocks_plus_lipids"
FINAL_RUN = RUNS / f"semaglutide__{RUN_SUFFIX}__modalities_paper__tabular"
EXPOSURE_RUNS = {
    "any_glp1": RUNS / f"any_glp1__{RUN_SUFFIX}__modalities_sleep-anthropometrics-blood_pressure__tabular",
    "non_semaglutide": RUNS / f"non_semaglutide__{RUN_SUFFIX}__modalities_sleep-anthropometrics-blood_pressure__tabular",
}
OUT = RUN / "outputs/unified_cohort/sleep_robustness"

REM = "heart_rate_mean_during_rem"
REM_MINUS_NREM = "heart_rate_rem_minus_nrem"
SLEEP_HEART_RATE = [
    "heart_rate_mean_during_sleep",
    "heart_rate_mean_during_wake",
    "heart_rate_mean_during_nrem",
    REM,
    REM_MINUS_NREM,
]
# The endpoints the visit-transition paper tested across designs and exposure definitions.
DESIGN_ENDPOINTS = {
    "anthropometrics": ["weight"],
    "blood_pressure": ["sitting_blood_pressure_systolic", "sitting_blood_pressure_pulse_rate"],
    "sleep": SLEEP_HEART_RATE,
}
EXPOSURE_ENDPOINTS = {
    "anthropometrics": ["weight"],
    "blood_pressure": ["sitting_blood_pressure_systolic"],
    "sleep": [REM, REM_MINUS_NREM],
}
# The ten endpoints of Supplementary Figure S4.
DRIFT_ENDPOINTS = {
    "blood_pressure": [
        "sitting_blood_pressure_systolic",
        "sitting_blood_pressure_diastolic",
        "sitting_blood_pressure_pulse_rate",
    ],
    "sleep": [
        "heart_rate_mean_during_sleep",
        "heart_rate_mean_during_wake",
        "heart_rate_mean_during_nrem",
        REM,
    ],
    "dxa": ["body_comp_total_fat_mass", "body_comp_total_lean_mass", "total_scan_vat_area"],
}
RESAMPLING_SEED_OFFSET = 29
DEFAULT_N_BOOTSTRAP = 500


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Pipeline:
    """The step 26/29/37/48/50 modules plus the cohort inputs every rebuild shares."""

    def __init__(self) -> None:
        self.step48 = load_module("robust_step48", RUN / "steps/48_unified_cohort.py")
        self.step26 = self.step48._load("robust_step26", "26_paper_reanalysis.py")
        self.step31 = self.step48._load("robust_step31", "31_raw_sleep_glp_cohort.py")
        self.step47 = self.step48._load("robust_step47", "47_date_anchored_pairing.py")
        self.step29 = load_module("robust_step29", RUN / "steps/29_sleep_bootstrap_rematching.py")
        self.step37 = load_module(
            "robust_step37", RUN / "steps/37_relative_scale_sleep_sensitivity.py")
        self.step50 = load_module(
            "robust_step50", RUN / "steps/50_unified_alternative_matching.py")
        self.matching = importlib.import_module("research_utils.analysis.models")
        self.step26.SUPPORTED_STAGES = tuple(self.step48.MAIN_STAGES)
        self.exact_cols = self.step48.EXACT_BLOCKS[self.step48.DEFAULT_BLOCKING]

        self.covariates = self.step26.load_covariates()
        all_glp = self.step48.load_all_glp_reports()
        self.excluded = set(self.step48.as_registration_code(
            pd.Series(sorted(all_glp.participant.unique())), self.step26))
        self.canonical_dates = self.step48.canonical_visit_dates(self.step26)
        self._exposures: dict[str, pd.DataFrame] = {}
        self._panels: dict[str, tuple[pd.DataFrame, pd.Series]] = {}

    def exposure(self, key: str) -> pd.DataFrame:
        if key not in self._exposures:
            frame = self.step48.build_exposure(self.step31, self.step48.EXPOSURE_SETS[key])
            frame["participant_id"] = self.step48.as_registration_code(
                frame.participant, self.step26).to_numpy()
            self._exposures[key] = frame
        return self._exposures[key]

    def panel(self, modality: str) -> tuple[pd.DataFrame, pd.Series]:
        # No panel in this step depends on the exposure argument (only diet does), so the
        # primary exposure is passed and the panel is shared across exposure definitions.
        if modality not in self._panels:
            values = self.step48.load_panel(self.step26, modality, self.exposure("semaglutide"))
            dates = self.step48.dates_for(self.step26, modality, self.canonical_dates)
            self._panels[modality] = (values, dates)
        return self._panels[modality]

    def rebuild(self, modality: str, exposure_key: str, reference_run: Path) -> dict:
        """Reconstruct one step-48 match and prove it is the versioned one."""
        values, dates = self.panel(modality)
        outcome, windows, _audit = self.step48.run_instrument(
            self.step26, self.step31, self.step47, modality, values,
            self.exposure(exposure_key), dates, self.covariates, self.excluded,
            self.step48.FLOOR_MONTHS, self.step48.DEFAULT_BLOCKING,
            stages=self.step48.MAIN_STAGES,
        )
        if not outcome or outcome["result"].empty:
            raise RuntimeError(f"{exposure_key}/{modality}: match could not be rebuilt")
        assert_matches_run(outcome["result"], reference_run / f"did_{modality}.csv",
                           f"{exposure_key}/{modality}")
        outcome["values"] = values
        outcome["windows"] = windows
        outcome["deltas"] = self.step26.unit_deltas(values, outcome["units"])
        return outcome


def assert_matches_run(rebuilt: pd.DataFrame, saved_path: Path, label: str) -> None:
    saved = pd.read_csv(saved_path).set_index("measurement")
    rebuilt = rebuilt.set_index("measurement")
    shared = saved.index.intersection(rebuilt.index)
    if len(shared) != len(saved):
        raise AssertionError(f"{label}: rebuilt endpoints differ from {saved_path}")
    if not np.array_equal(saved.loc[shared, "n_pairs"].to_numpy(int),
                          rebuilt.loc[shared, "n_pairs"].to_numpy(int)):
        raise AssertionError(f"{label}: rebuilt pair counts differ from {saved_path}")
    if not np.allclose(saved.loc[shared, "effect"].to_numpy(float),
                       rebuilt.loc[shared, "effect"].to_numpy(float), rtol=0, atol=1e-12):
        raise AssertionError(f"{label}: rebuilt estimates differ from {saved_path}")


def design_estimates(pipe: Pipeline, outcome: dict, modality: str,
                     endpoints: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Primary 1:1, capped reuse and exact-stratum overlap estimates for `endpoints`."""
    step26, step50, matching = pipe.step26, pipe.step50, pipe.matching
    units, deltas, primary_pairs = outcome["units"], outcome["deltas"], outcome["pairs"]
    eligible = int(units.loc[units.treated.eq(1), "participant_id"].nunique())

    capped_pairs = matching.capacity_caliper_match(
        units, covariate_cols=step26.MATCH_COVARIATES, treatment_col="treated",
        id_col="participant_id", exact_cols=pipe.exact_cols, max_ratio=3,
        control_capacity=2, caliper_sd=pipe.step48.CALIPER_SD, seed=pipe.step48.SEED,
    )
    diagnostics = []
    for design, pairs in [(step50.DESIGN_PRIMARY, primary_pairs),
                          (step50.DESIGN_CAPPED, capped_pairs)]:
        balance = step50.balance_from_pairs(step26, units, pairs, modality, design)
        ess_treated, ess_controls = step50.pair_ess(pairs)
        diagnostics.append(step50.diagnostic_row(
            modality=modality, design=design, eligible_treated=eligible,
            matched_treated=int(pairs.treated_id.nunique()),
            unique_controls=int(pairs.control_id.nunique()),
            ess_treated=ess_treated, ess_controls=ess_controls,
            max_control_reuse=int(pairs.groupby("control_id").size().max()),
            max_abs_smd=float(balance.smd_after.abs().max()),
            block_difference=step50.pair_block_distribution_difference(pairs, pipe.exact_cols),
            estimand="retained treated initiators (ATT-like)",
        ))
    diagnostics = pd.DataFrame(diagnostics)
    design_gate = diagnostics.set_index("design").design_gate_pass

    primary = step50.add_pair_endpoint_balance(
        step26, primary_pairs, deltas, step50.primary_effects(outcome["result"]))
    capped = step50.add_pair_endpoint_balance(
        step26, capped_pairs, deltas,
        step26.weighted_effect(capped_pairs, deltas, design=step50.DESIGN_CAPPED,
                               modality=modality))
    for frame, design in [(primary, step50.DESIGN_PRIMARY), (capped, step50.DESIGN_CAPPED)]:
        frame["gate_pass"] = bool(design_gate.loc[design])
    overlap_units = step50.exact_stratum_overlap(matching, step26, units, pipe.exact_cols)
    overlap = step50.overlap_effects(step26, overlap_units, deltas, modality)
    overlap["gate_pass"] = overlap.endpoint_balance_pass.astype(bool)

    columns = ["modality", "measurement", "design", "n_treated", "n_controls", "effect",
               "ci_low", "ci_high", "p", "gate_pass"]
    estimates = pd.concat([primary, capped, overlap], ignore_index=True)
    estimates = estimates.loc[estimates.measurement.isin(endpoints), columns]
    if estimates.groupby("measurement").design.nunique().lt(3).any():
        raise AssertionError(f"{modality}: an endpoint is missing a design estimate")
    return estimates, diagnostics


def replicate_estimates(pair_rows: pd.DataFrame, replicate: int) -> list[dict]:
    records = []
    for measurement in SLEEP_HEART_RATE:
        group = pair_rows.loc[pair_rows.measurement.eq(measurement), ["treated_id", "did"]]
        group = group.dropna()
        records.append({
            "replicate": replicate, "estimand_type": "endpoint", "estimand": measurement,
            "n_pairs": int(group.treated_id.nunique()),
            "estimate": float(group.did.mean()) if len(group) else np.nan,
        })
    return records


def rematched_bootstrap(pipe: Pipeline, sleep: dict, n_bootstrap: int) -> pd.DataFrame:
    """Percentile intervals when the propensity model and match are refit per replicate."""
    step26, step48, matching = pipe.step26, pipe.step48, pipe.matching
    base_units = sleep["units"].drop(columns=["propensity", "ps_logit"])
    observed = sleep["result"].set_index("measurement").loc[SLEEP_HEART_RATE]
    observed = pd.DataFrame({
        "estimand_type": "endpoint", "estimand": SLEEP_HEART_RATE,
        "observed_estimate": observed.effect.to_numpy(float),
        "observed_n_pairs": observed.n_pairs.to_numpy(int),
        "fixed_match_ci_low": observed.ci_low.to_numpy(float),
        "fixed_match_ci_high": observed.ci_high.to_numpy(float),
    })
    rng = np.random.default_rng(step48.SEED + RESAMPLING_SEED_OFFSET)
    records: list[dict] = []
    for replicate in range(n_bootstrap):
        units, deltas = pipe.step29.resample_treated_units(base_units, sleep["deltas"], rng)
        units = matching.propensity_score(
            units, treatment_col="treated", covariate_cols=step26.MATCH_COVARIATES)
        pairs = matching.optimal_pair_match(
            units, covariate_cols=step26.MATCH_COVARIATES, treatment_col="treated",
            id_col="participant_id", exact_cols=pipe.exact_cols,
            caliper_sd=step48.CALIPER_SD, seed=step48.SEED + replicate,
        )
        if pairs.empty:
            continue
        records.extend(replicate_estimates(step26.pair_effects(pairs, deltas), replicate))
        if (replicate + 1) % 25 == 0:
            logging.info("rematched bootstrap: %d/%d replicates", replicate + 1, n_bootstrap)
    summary = pipe.step29.summarize(pd.DataFrame(records), observed, n_bootstrap)
    fixed = observed.set_index("estimand")[["fixed_match_ci_low", "fixed_match_ci_high"]]
    return summary.merge(fixed, left_on="estimand", right_index=True, how="left")


def untreated_drift(pipe: Pipeline, outcome: dict, modality: str,
                    endpoints: list[str]) -> pd.DataFrame:
    drift = pipe.step26.untreated_drift(
        outcome["deltas"], outcome["units"], outcome["pairs"], modality)
    arms = outcome["result"].set_index("measurement")[[
        "n_pairs", "control_delta", "control_delta_ci_low", "control_delta_ci_high",
    ]]
    drift = drift.merge(arms, left_on="measurement", right_index=True, how="inner")
    missing = set(endpoints) - set(drift.measurement)
    if missing:
        raise AssertionError(f"{modality}: no drift reference for {sorted(missing)}")
    return drift.loc[drift.measurement.isin(endpoints)]


def relative_scale(pipe: Pipeline, sleep: dict) -> dict[str, pd.DataFrame]:
    """Step 37's relative-scale estimators applied to the unified sleep pair rows."""
    step26, step37 = pipe.step26, pipe.step37
    rows = sleep["rows"]
    log_rows = step37.log_scale_pair_rows(rows)
    contrasts = step37.relative_contrasts(
        log_rows, contrast_model=step26.contrast_model, bh_fdr=step26.bh_fdr)
    dependence = step37.baseline_dependence(rows, fit_hc3=step26.fit_hc3, min_n=step26.MIN_N)
    systemic = step37.systemic_rise_adjustment(
        log_rows, fit_hc3=step26.fit_hc3, min_n=step26.MIN_N)

    # Per-pair values behind the figure's point clouds: no identifiers, and sorted so the
    # row order carries no link back to a pair.
    absolute = step37._gap_frame(log_rows, "did")
    relative = step37._gap_frame(log_rows, "log_did")
    per_pair = pd.DataFrame({
        "rem_minus_nrem_did": absolute.gap,
        "ratio_of_ratios": np.exp(relative.gap),
        "wake_did": absolute.wake,
    }).dropna().sort_values("rem_minus_nrem_did").reset_index(drop=True)

    # The same three stage/state contrasts on the absolute (bpm) scale, paired within
    # participant and corrected together, as in the relative-scale family above.
    absolute_contrasts = pd.DataFrame([
        {**step26.contrast_model(rows, step37.heart_rate_endpoint(a),
                                 step37.heart_rate_endpoint(b),
                                 f"{step37.STATE_LABEL[a]} minus {step37.STATE_LABEL[b]} heart-rate DiD"),
         "state_a": a, "state_b": b}
        for a, b in step37.CONTRAST_PAIRS
    ])
    absolute_contrasts["fdr_within_contrast_family"] = step26.bh_fdr(
        absolute_contrasts.p.to_numpy(float))

    reported = rows.loc[rows.measurement.eq(REM_MINUS_NREM), "did"].dropna()
    if len(per_pair) != len(reported) or not np.isclose(
            per_pair.rem_minus_nrem_did.mean(), reported.mean(), rtol=0, atol=1e-9):
        raise AssertionError("per-pair REM minus non-REM gap does not reproduce the contrast")
    return {"relative_scale_contrasts": contrasts, "baseline_dependence": dependence,
            "systemic_rise_adjustment": systemic, "relative_scale_per_pair": per_pair,
            "absolute_stage_contrasts": absolute_contrasts}


def pair_rows_for(pipe: Pipeline, outcome: dict, values: pd.DataFrame) -> pd.DataFrame:
    """Another panel's DiD rows on exactly the pairs of `outcome` (same units, same controls)."""
    return pipe.step26.pair_effects(
        outcome["pairs"], pipe.step26.unit_deltas(values, outcome["units"]))


def weight_attenuation(pipe: Pipeline, sleep: dict) -> pd.DataFrame:
    """Does the pair's own weight change explain its REM heart-rate change?"""
    weight_rows = pair_rows_for(pipe, sleep, pipe.panel("anthropometrics")[0])
    return pd.DataFrame([
        pipe.step26.attenuation_model(sleep["rows"], weight_rows, measurement, "weight",
                                      f"sleep:{measurement} adjusted for pairwise weight change")
        for measurement in [REM, REM_MINUS_NREM]
    ])


def rem_composition(pipe: Pipeline, sleep: dict) -> pd.DataFrame:
    """Change in sleep architecture on the same pairs, and whether the REM gap tracks it."""
    composition_rows = pair_rows_for(
        pipe, sleep, pipe.step37.load_sleep_composition(pipe.step26))
    return pipe.step37.rem_composition_check(
        pipe.step37.log_scale_pair_rows(sleep["rows"]), composition_rows,
        fit_hc3=pipe.step26.fit_hc3, min_n=pipe.step26.MIN_N)


def duration_slopes(pipe: Pipeline, outcomes: dict[str, dict],
                    endpoints: dict[str, list[str]]) -> pd.DataFrame:
    """Slope of each pair's DiD on the initiator's months on drug at the post visit."""
    records = []
    for modality, measurements in endpoints.items():
        outcome = outcomes[modality]
        months = outcome["windows"].set_index("RegistrationCode").months_on_drug
        for measurement in measurements:
            group = outcome["rows"].loc[outcome["rows"].measurement.eq(measurement)]
            group = group.assign(months=group.treated_id.map(months)).dropna(
                subset=["did", "months"])
            x = np.column_stack([np.ones(len(group)), group.months.to_numpy(float)])
            beta, standard_error, p_value = pipe.step26.fit_hc3(group.did.to_numpy(float), x)
            records.append({
                "modality": modality, "measurement": measurement, "n_pairs": len(group),
                "months_median": float(group.months.median()),
                "months_min": float(group.months.min()),
                "months_max": float(group.months.max()),
                "slope_per_month": float(beta[1]),
                "slope_ci_low": float(beta[1] - 1.96 * standard_error[1]),
                "slope_ci_high": float(beta[1] + 1.96 * standard_error[1]),
                "slope_p": float(p_value[1]),
            })
    frame = pd.DataFrame(records)
    frame["fdr"] = pipe.step26.bh_fdr(frame.slope_p.to_numpy(float))
    return frame


def exposure_estimates(pipe: Pipeline, outcome: dict, exposure_key: str,
                       modality: str) -> pd.DataFrame:
    """1:1 and exact-stratum overlap estimates for one exposure definition."""
    step26, step50 = pipe.step26, pipe.step50
    endpoints = EXPOSURE_ENDPOINTS[modality]
    primary = step50.primary_effects(outcome["result"])
    primary["gate_pass"] = bool(outcome["result"].design_gate_pass.iloc[0])
    primary["design"] = "optimal_1to1"
    overlap_units = step50.exact_stratum_overlap(
        pipe.matching, step26, outcome["units"], pipe.exact_cols)
    overlap = step50.overlap_effects(step26, overlap_units, outcome["deltas"], modality)
    overlap["gate_pass"] = overlap.endpoint_balance_pass.astype(bool)
    overlap["design"] = "exact_stratum_overlap"
    frame = pd.concat([primary, overlap], ignore_index=True)
    frame = frame.loc[frame.measurement.isin(endpoints), [
        "modality", "measurement", "design", "n_treated", "effect", "ci_low", "ci_high",
        "p", "gate_pass"]]
    frame.insert(0, "exposure", exposure_key)
    frame["exposed_participants"] = len(pipe.exposure(exposure_key))
    frame["max_abs_smd_1to1"] = float(outcome["max_abs_smd"])
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    for path in [FINAL_RUN, *EXPOSURE_RUNS.values()]:
        if not (path / "run_manifest.json").exists():
            raise FileNotFoundError(f"missing versioned step-48 run: {path}")

    pipe = Pipeline()
    primary = {modality: pipe.rebuild(modality, "semaglutide", FINAL_RUN)
               for modality in ["sleep", "blood_pressure", "anthropometrics", "dxa"]}

    designs, diagnostics = zip(*[
        design_estimates(pipe, primary[modality], modality, endpoints)
        for modality, endpoints in DESIGN_ENDPOINTS.items()
    ])
    pd.concat(designs).to_csv(OUT / "design_estimates.csv", index=False)
    pd.concat(diagnostics).to_csv(OUT / "design_diagnostics.csv", index=False)

    drift = pd.concat([untreated_drift(pipe, primary[modality], modality, endpoints)
                       for modality, endpoints in DRIFT_ENDPOINTS.items()])
    drift.to_csv(OUT / "untreated_drift.csv", index=False)

    for name, frame in relative_scale(pipe, primary["sleep"]).items():
        frame.to_csv(OUT / f"{name}.csv", index=False)

    exposures = [exposure_estimates(pipe, primary[modality], "semaglutide", modality)
                 for modality in EXPOSURE_ENDPOINTS]
    for exposure_key, reference_run in EXPOSURE_RUNS.items():
        for modality in EXPOSURE_ENDPOINTS:
            outcome = pipe.rebuild(modality, exposure_key, reference_run)
            exposures.append(exposure_estimates(pipe, outcome, exposure_key, modality))
    pd.concat(exposures).to_csv(OUT / "exposure_definitions.csv", index=False)

    weight_attenuation(pipe, primary["sleep"]).to_csv(
        OUT / "weight_attenuation.csv", index=False)
    rem_composition(pipe, primary["sleep"]).to_csv(OUT / "rem_composition.csv", index=False)
    duration_slopes(pipe, primary, {
        "sleep": [REM, REM_MINUS_NREM],
        "blood_pressure": ["sitting_blood_pressure_systolic"],
    }).to_csv(OUT / "duration_slopes.csv", index=False)

    bootstrap = rematched_bootstrap(pipe, primary["sleep"], args.n_bootstrap)
    bootstrap.to_csv(OUT / "rematched_bootstrap_summary.csv", index=False)

    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "final_tabular_run": str(FINAL_RUN.relative_to(RUN)),
        "final_tabular_manifest_sha256": sha256(FINAL_RUN / "run_manifest.json"),
        "exposure_runs": {key: str(path.relative_to(RUN)) for key, path in EXPOSURE_RUNS.items()},
        "exposure_run_manifest_sha256": {
            key: sha256(path / "run_manifest.json") for key, path in EXPOSURE_RUNS.items()},
        "step48_sha256": sha256(RUN / "steps/48_unified_cohort.py"),
        "clean_dated_semaglutide_starters": int(len(pipe.exposure("semaglutide"))),
        "sleep_matched": int(len(primary["sleep"]["matched"])),
        "n_bootstrap_requested": args.n_bootstrap,
        "resampling_seed": int(pipe.step48.SEED + RESAMPLING_SEED_OFFSET),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
