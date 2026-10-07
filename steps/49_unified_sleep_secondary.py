"""Rerun sleep-specific secondary models on the final unified cohort.

This step deliberately rebuilds the sleep match through ``steps/48_unified_cohort.py``
and checks it against the versioned final tabular run before fitting anything.  It writes
aggregate estimates only; participant identifiers and matched-pair rows stay in memory.

Outputs
-------
``outputs/unified_cohort/secondary_sleep/``
    hrv_rate_adjustment.csv
        Pair-level HRV DiD models adjusted for the simultaneous state-specific heart-rate
        DiD.  The intercept is the estimated HRV change at zero heart-rate change.
    raw_stage_results.csv
        Raw-channel 1:1 DiD estimates for REM, light, deep, wake and prespecified
        within-night stage contrasts.
    raw_stage_balance.csv
        Covariate balance for the raw-channel match.
    raw_stage_design.csv
        Retention and balance-gate status.  This must accompany every use of the raw result.
    manifest.json
        Input run and script hashes plus the cohort checks used to prevent drift.
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


RUN = Path(__file__).resolve().parents[1]
FINAL_RUN = (
    RUN
    / "outputs/unified_cohort/runs/"
    / "semaglutide__floor_3m__stages_main__blocks_plus_lipids__modalities_paper__tabular"
)
OUT = RUN / "outputs/unified_cohort/secondary_sleep"

HRV_MODELS = [
    ("neurokit_hrv_time_pnn20_during_rem", "heart_rate_mean_during_rem"),
    ("neurokit_hrv_time_rmssd_during_rem", "heart_rate_mean_during_rem"),
    ("neurokit_hrv_time_sdnn_during_rem", "heart_rate_mean_during_rem"),
    ("neurokit_hrv_time_pnn20_during_nrem", "heart_rate_mean_during_nrem"),
    ("neurokit_hrv_time_rmssd_during_nrem", "heart_rate_mean_during_nrem"),
    ("neurokit_hrv_time_sdnn_during_nrem", "heart_rate_mean_during_nrem"),
]

RAW_MEASUREMENTS = {
    "rem": "heart_rate_mean_during_rem",
    "light": "heart_rate_mean_during_light",
    "deep": "heart_rate_mean_during_deep",
    "nrem": "heart_rate_mean_during_nrem",
    "wake": "heart_rate_mean_during_wake",
    "sleep": "heart_rate_mean_during_sleep",
    "rem_minus_nrem": "heart_rate_rem_minus_nrem",
    "rem_minus_light": "heart_rate_rem_minus_light",
    "rem_minus_deep": "heart_rate_rem_minus_deep",
    "deep_minus_light": "heart_rate_deep_minus_light",
}


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


def raw_panel(step48, step31, link: pd.DataFrame, needed: pd.DataFrame) -> pd.DataFrame:
    """Return raw visit summaries in the same long schema as the tabular sleep panel."""
    means = step48.raw_visit_means(step31, needed)
    if means.empty:
        return pd.DataFrame(
            columns=["participant_id", "research_stage", "measurement", "value"]
        )
    frame = link.merge(means, on="uuid", how="inner")
    sleep_seconds = frame[["rem_seconds", "light_seconds", "deep_seconds"]].sum(axis=1)
    weighted = (
        frame.rem.fillna(0) * frame.rem_seconds
        + frame.light.fillna(0) * frame.light_seconds
        + frame.deep.fillna(0) * frame.deep_seconds
    )
    frame["sleep"] = np.where(
        sleep_seconds > 0, weighted / sleep_seconds.replace(0, np.nan), np.nan
    )
    frame["rem_minus_nrem"] = frame.rem - frame.nrem
    frame["rem_minus_light"] = frame.rem - frame.light
    frame["rem_minus_deep"] = frame.rem - frame.deep
    frame["deep_minus_light"] = frame.deep - frame.light
    long = frame.melt(
        id_vars=["participant_id", "research_stage"],
        value_vars=list(RAW_MEASUREMENTS),
        var_name="state",
        value_name="value",
    ).dropna(subset=["value"])
    long["measurement"] = long.state.map(RAW_MEASUREMENTS)
    return long[["participant_id", "research_stage", "measurement", "value"]]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    step48_path = RUN / "steps/48_unified_cohort.py"
    step48 = load_module("step48_unified_secondary", step48_path)
    step26 = step48._load("step26_secondary", "26_paper_reanalysis.py")
    step31 = step48._load("step31_secondary", "31_raw_sleep_glp_cohort.py")
    step47 = step48._load("step47_secondary", "47_date_anchored_pairing.py")
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

    values = step26.load_snapshot_panel("sleep")
    dates = step48.visit_dates_for(step26, "sleep")
    tabular, windows, _ = step48.run_instrument(
        step26,
        step31,
        step47,
        "sleep",
        values,
        exposure,
        dates,
        covariates,
        excluded,
        step48.FLOOR_MONTHS,
        step48.DEFAULT_BLOCKING,
        stages=step48.MAIN_STAGES,
    )
    if not tabular:
        raise RuntimeError("The unified sleep cohort could not be reconstructed")

    # Guard against a hidden configuration change: the rebuilt sleep estimates have to be
    # the exact estimates already approved in the final versioned run.
    saved = pd.read_csv(FINAL_RUN / "did_sleep.csv").set_index("measurement")
    rebuilt = tabular["result"].set_index("measurement")
    shared = saved.index.intersection(rebuilt.index)
    if not np.array_equal(
        saved.loc[shared, "n_pairs"].to_numpy(int),
        rebuilt.loc[shared, "n_pairs"].to_numpy(int),
    ):
        raise AssertionError("Rebuilt sleep pair counts differ from the final tabular run")
    if not np.allclose(
        saved.loc[shared, "effect"].to_numpy(float),
        rebuilt.loc[shared, "effect"].to_numpy(float),
        rtol=0,
        atol=1e-12,
    ):
        raise AssertionError("Rebuilt sleep estimates differ from the final tabular run")

    rate_models = []
    for outcome, covariate in HRV_MODELS:
        state = "rem" if outcome.endswith("_during_rem") else "nrem"
        rate_models.append(
            step26.attenuation_model(
                tabular["rows"],
                tabular["rows"],
                outcome,
                covariate,
                f"sleep:{outcome} adjusted for pairwise {state} heart-rate change",
            )
        )
    rate_models = pd.DataFrame(rate_models)
    if rate_models["covariate_adjusted_intercept"].isna().any():
        raise AssertionError("At least one HRV rate-adjustment model was not estimable")
    rate_models.to_csv(OUT / "hrv_rate_adjustment.csv", index=False)

    link = step31.load_sleep_visits()
    link["participant_id"] = step48.as_registration_code(link.participant, step26).to_numpy()
    link["research_stage"] = link.research_stage.map(step48.canonical_stage)
    inherited = windows[windows.RegistrationCode.isin(tabular["matched"])]
    wanted = set(tabular["units"].participant_id)
    needed = step48.visits_to_extract(link, wanted, inherited)
    raw_values = raw_panel(step48, step31, link, needed)
    raw, raw_windows, _ = step48.run_instrument(
        step26,
        step31,
        step47,
        "sleep_raw",
        raw_values,
        exposure,
        dates,
        covariates,
        excluded,
        step48.FLOOR_MONTHS,
        step48.DEFAULT_BLOCKING,
        windows=inherited,
        stages=step48.MAIN_STAGES,
    )
    if not raw or raw["result"].empty:
        raise RuntimeError("The unified raw-channel cohort was not estimable")
    if not raw["matched"].issubset(tabular["matched"]):
        raise AssertionError("Raw-channel treated participants are not nested in tabular match")
    raw["result"].to_csv(OUT / "raw_stage_results.csv", index=False)
    raw["balance"].to_csv(OUT / "raw_stage_balance.csv", index=False)

    raw_design_pass = bool(
        raw["retention"] >= step26.MIN_MATCH_RETENTION
        and raw["max_abs_smd"] < step26.MAX_ABS_SMD
    )
    raw_design = pd.DataFrame(
        [
            {
                "pairable": len(raw_windows),
                "eligible_treated": len(raw["eligible"]),
                "matched": len(raw["matched"]),
                "retention": raw["retention"],
                "max_abs_smd": raw["max_abs_smd"],
                "design_gate_pass": raw_design_pass,
            }
        ]
    )
    raw_design.to_csv(OUT / "raw_stage_design.csv", index=False)

    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "final_tabular_run": str(FINAL_RUN.relative_to(RUN)),
        "final_tabular_manifest_sha256": sha256(FINAL_RUN / "run_manifest.json"),
        "step48_sha256": sha256(step48_path),
        "script_sha256": sha256(Path(__file__)),
        "clean_dated_semaglutide_starters": int(len(exposure)),
        "sleep_pairable": int(len(windows)),
        "sleep_eligible": int(len(tabular["eligible"])),
        "sleep_matched": int(len(tabular["matched"])),
        "raw_matched": int(len(raw["matched"])),
        "raw_max_abs_smd": float(raw["max_abs_smd"]),
        "raw_design_gate_pass": raw_design_pass,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(rate_models.to_string(index=False))
    print("\nRaw-channel design")
    print(raw_design.to_string(index=False))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
